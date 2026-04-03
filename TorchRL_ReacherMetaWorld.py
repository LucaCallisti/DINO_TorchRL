import os
os.environ['MUJOCO_GL'] = "egl"
os.environ['PYOPENGL_PLATFORM'] = "egl"

import torch
import torch.nn as nn
from torch import optim
import metaworld

from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    Compose, InitTracker, StepCounter, TransformedEnv, 
    RewardSum, DoubleToFloat
)
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.modules.models import MLP
from torchrl.modules import ValueOperator
from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
from torchrl.trainers.algorithms import SACTrainer
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers import WandbLogger
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.envs import ParallelEnv

import tempfile
import wandb

# =========================
# CONFIG CENTRALIZZATO
# =========================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Environment
    task_name = "reach-v3"
    render_mode = "rgb_array"
    camera_name = "corner2"
    max_steps = 500
    num_parallel_envs = 4
    
    # Network
    hidden_layers = [512, 512, 256]
    
    # Training
    total_frames = 1_000_000
    total_frames_per_task = {
        "reach-v3":         300_000,
        "push-v3":          500_000,
        "pick-place-v3":    800_000,
        "door-open-v3":     500_000,
        "drawer-open-v3":   400_000,
        "drawer-close-v3":  300_000,
        "button-press-topdown-v3": 500_000,
        "peg-insert-side-v3":     800_000,
        "window-open-v3":   400_000,
        "window-close-v3":  400_000,
        # default per task non listati
        "default":          1_000_000,
    }
    frames_per_batch = 100
    init_random_frames = 10_000
    
    # Replay Buffer
    buffer_size = 1_000_000
    batch_size = 256
    
    # SAC
    alpha_init = 1.0
    gamma = 0.99
    tau = 0.005
    lr = 3e-4
    optim_steps_per_batch = 10
    
    # Logging
    log_interval = 1000
    test_interval = 10_000
    checkpoint_interval = 50_000
    test_num_episodes = 3
    wandb_project = "MetaWorld"
    wandb_entity = "Torch_RL"
    wandb_exp_name = "Reacher"


# =========================
# CUSTOM VIDEO RECORDERS
# =========================
class CopyingVideoRecorder(VideoRecorder):
    
    def _apply_transform(self, observation):
        if hasattr(observation, 'data'):
            import numpy as np
            observation.data = np.array(observation.data).copy()
        if observation.data is None:
            print("[WARNING] Observation data is None in CopyingVideoRecorder.")
        if observation.data.min() < 0 or observation.data.max() > 255:
            print(f"[WARNING] Observation data has unexpected range: min={observation.data.min()}, max={observation.data.max()}")
        if observation.data.dtype != np.uint8:
            print(f"[WARNING] Observation data has unexpected dtype: {observation.data.dtype}")
        
        return super()._apply_transform(observation)


class SelectiveVideoRecorder(VideoRecorder):
    
    def __init__(self, *args, record_every_n_episodes=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_every_n_episodes = record_every_n_episodes
        self.episode_count = 0
        self.should_record = False
    
    def _reset(self, tensordict, tensordict_reset):
        self.episode_count += 1
        self.should_record = (self.episode_count % self.record_every_n_episodes == 0)
        return super()._reset(tensordict, tensordict_reset)
    
    def _apply_transform(self, observation):
        if self.should_record:
            if hasattr(observation, 'data'):
                import numpy as np
                observation.data = np.array(observation.data).copy()
            return super()._apply_transform(observation)
        return observation


# =========================
# ENVIRONMENT SETUP
# =========================
def setup_env(cfg):
    ml1 = metaworld.ML1(cfg.task_name)
    task = ml1.train_tasks[0]
    def make_env():
        gym_env = ml1.train_classes[cfg.task_name](
            render_mode=cfg.render_mode,
            camera_name=cfg.camera_name
        )
        task = ml1.train_tasks[0]
        gym_env.unwrapped.set_task(task)
        return GymWrapper(gym_env, device=cfg.device)
    base_env = ParallelEnv(
        num_workers=cfg.num_parallel_envs,
        create_env_fn=make_env,
        # device=cfg.device,
        device="cpu"
    )
        
    transforms = [
        InitTracker(),
        StepCounter(max_steps=cfg.max_steps),
        RewardSum(in_keys=["reward"], out_keys=["reward_sum"]),
        DoubleToFloat(in_keys=["observation"]),
    ]
    
    env = TransformedEnv(base_env, Compose(*transforms)).to(cfg.device)
    return env, task, ml1

def setup_test_env(cfg, task, ml1, logger):
    gym_env = ml1.train_classes[cfg.task_name](
        render_mode=cfg.render_mode,
        camera_name=cfg.camera_name
    )
    gym_env.unwrapped.set_task(task)
    
    transforms = [
        StepCounter(max_steps=cfg.max_steps, truncated_key="truncated", step_count_key="step_count"),
        DoubleToFloat(in_keys=["observation"]),
        PixelRenderTransform(out_keys=["pixels"], as_non_tensor=True),
        CopyingVideoRecorder(logger=logger, tag="test/video", in_keys=["pixels"]),
    ]
    
    test_env = TransformedEnv(
        GymWrapper(gym_env, device=cfg.device),
        Compose(*transforms))
    # ).to(cfg.device)
    
    return test_env


# =========================
# NETWORK SETUP
# =========================
def setup_networks(cfg, obs_dim, action_dim, action_spec):
    # Actor
    policy_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            activation_class=nn.ReLU,
            out_features=2 * action_dim,
            num_cells=cfg.hidden_layers
        ),
        NormalParamExtractor()
    )
    policy_td_module = TensorDictModule(
        module=policy_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )
    
    actor = ProbabilisticActor(
        module=policy_td_module,
        spec=action_spec, 
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ).to(cfg.device)
    
    # Critic
    q_net = MLP(
        in_features=obs_dim + action_dim,
        out_features=1,
        depth=2,
        num_cells=256,
    ).to(cfg.device)
    
    qvalue = ValueOperator(
        module=q_net,
        in_keys=["observation", "action"],
    ).to(cfg.device)
    
    return actor, qvalue


# =========================
# TRAINING HOOKS
# =========================
class VideoLogHook:    
    def __init__(self, trainer, env, interval=1):
        self.trainer = trainer
        self.env = env
        self.interval = interval
    
    def __call__(self, batch):
        current_step = self.trainer.collected_frames
        if current_step % self.interval == 0:
            for transform in self.env.transform:
                if isinstance(transform, VideoRecorder):
                    transform.dump(step=current_step)
        return {}

class EpisodeRewardHook:    
    def __init__(self, trainer, logger):
        self.trainer = trainer
        self.logger = logger
    
    def __call__(self, batch):
        dones = batch.get(("next", "done"), None)
        if dones is not None and dones.any():
            reward_sums = batch.get(("next", "reward_sum"), None)
            if reward_sums is not None:
                final_rewards = reward_sums[dones]
                if final_rewards.numel() > 0:
                    avg_reward = final_rewards.mean().item()
                    current_step = self.trainer.collected_frames
                    self.logger.log_scalar(name="train/episode_final_reward", value=avg_reward, step=current_step)
        return {}

class TestEvaluationHook:    
    def __init__(self, trainer, test_env, actor, logger, cfg):
        self.trainer = trainer
        self.test_env = test_env
        self.actor = actor
        self.logger = logger
        self.interval = cfg.test_interval
        self.num_episodes = cfg.test_num_episodes
    
    def __call__(self, batch):
        current_step = self.trainer.collected_frames
        if current_step % self.interval == 0:
            self._run_test(current_step)
        return {}
    
    def _run_test(self, step):
        print(f"[TEST] Running evaluation at step {step}")
        rewards=[]
        for episode in range(self.num_episodes):
            rew = self._run_episode()
            rewards.append(rew)

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        current_step = self.trainer.collected_frames
        self.logger.log_scalar(name='test/avg_reward', value=avg_reward, step=current_step)

        # Dump video
        for transform in self.test_env.transform:
            if isinstance(transform, VideoRecorder):
                transform.dump(step=step)
    
    def _run_episode(self):
        td = self.test_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            with torch.no_grad():
                td = self.actor(td)
            td = self.test_env.step(td)

            episode_reward += td[("next", "reward")].item()
            terminated = td[("next", "done")].item() if ("next", "done") in td.keys(True) else False
            truncated = td[("next", "truncated")].item()
            done = terminated or truncated
            td = td["next"]
        return episode_reward



class CheckpointWandbHook:    
    def __init__(self, trainer, actor, qvalue, logger, cfg):
        self.trainer = trainer
        self.actor = actor
        self.qvalue = qvalue
        self.save_interval = cfg.checkpoint_interval
        self.temp_dir = tempfile.mkdtemp()
    
    def __call__(self, batch):
        current_step = self.trainer.collected_frames
        if current_step % self.save_interval == 0 and current_step > 0:
            actor_path = f"{self.temp_dir}/actor_step_{current_step}.pt"
            qvalue_path = f"{self.temp_dir}/qvalue_step_{current_step}.pt"
            
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.qvalue.state_dict(), qvalue_path)
            
            artifact = wandb.Artifact(f"checkpoint_step_{current_step}", type="model")
            artifact.add_file(actor_path, name="actor.pt")
            artifact.add_file(qvalue_path, name="qvalue.pt")
            wandb.run.log_artifact(artifact)
            
            
            os.remove(actor_path)
            os.remove(qvalue_path)
            
            print(f"[CHECKPOINT] Logged to W&B at step {current_step}")
        
        return {}


def move_batch_to_device(batch, device):
    return batch.to(device)



# =========================
# MAIN
# =========================
def main(cfg):
    
    # Setup ambienti
    print("Setting up training environment...")
    train_env, task, ml1 = setup_env(cfg)
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    action_dim = train_env.action_spec.shape[-1]
    
    logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        exp_name=cfg.wandb_exp_name,
        save_code=True,
        reinit=True
    )
    logger.log_hparams(vars(cfg))
    
    test_env = setup_test_env(cfg, task, ml1, logger)
    
    print("Setting up networks...")
    actor, qvalue = setup_networks(cfg, obs_dim, action_dim, train_env.action_spec)
    
    print("Setting up collector...")
    collector = Collector(
        train_env,
        actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        init_random_frames=cfg.init_random_frames,
        device=cfg.device,
    )
    
    # Setup replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer_size, device="cpu"),
        batch_size=cfg.batch_size,
    )
    
    # Setup loss
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        alpha_init=cfg.alpha_init,
        target_entropy=-action_dim,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    
    # Setup optimizer
    optimizer = optim.Adam(loss_module.parameters(), lr=cfg.lr)
    target_net_updater = SoftUpdate(loss_module, tau=cfg.tau)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = SACTrainer(
        collector=collector,
        total_frames=cfg.total_frames,
        replay_buffer=replay_buffer,
        frame_skip=1,
        optim_steps_per_batch=cfg.optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        log_interval=cfg.log_interval,
        target_net_updater=target_net_updater,
    )
    
    # Registration hooks
    trainer.register_op(
        "process_optim_batch",
        lambda batch: move_batch_to_device(batch, cfg.device)
    )
    
    trainer.register_op(
        "post_steps_log",
        VideoLogHook(trainer, train_env, interval=1)
    )
    
    trainer.register_op(
        "post_steps_log",
        TestEvaluationHook(trainer, test_env, actor, logger, cfg)
    )

    trainer.register_op(
        "post_steps_log", 
        EpisodeRewardHook(trainer, logger)
    )

    trainer.register_op(
        "post_steps_log",
        CheckpointWandbHook(trainer, actor, qvalue, logger, cfg)
    )

    
    # Training
    print("Starting training...")
    trainer.train()

    try:
        train_env.close()
    except Exception as e:
        print(f"Error closing training environment: {e}")
    try:
        wandb.finish()
    except Exception as e:
        print(f"Error finishing W&B run: {e}")


if __name__ == "__main__":
    cfg=Config()
    mt10 = metaworld.MT10()
    tasks = list(mt10.train_classes.keys())
    # dones=['reach-v3']
    dones=[]
    for task in tasks:
        if task not in dones:
            print(f"\nTraining on task: {task}\n")
            cfg.wandb_exp_name = task
            cfg.task_name = task
            cfg.total_frames = cfg.total_frames_per_task.get(task, cfg.total_frames_per_task["default"])
            main(cfg)