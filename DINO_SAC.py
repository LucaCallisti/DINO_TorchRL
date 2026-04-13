import os
os.environ['MUJOCO_GL'] = "egl"
os.environ['PYOPENGL_PLATFORM'] = "egl"

import numpy
import torch
import torch.nn as nn
from torch import optim
import metaworld

from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    Compose, InitTracker, StepCounter, TransformedEnv, 
    RewardSum, DoubleToFloat, FrameSkipTransform, CatFrames, Resize, UnsqueezeTransform
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

from wrappers import ToCHWTransform, SelectObservationTransform
from extractor import ExtractorTransform, DinoExtractor, Model
import numpy as np
import hydra
from omegaconf import OmegaConf

# =========================
# CUSTOM VIDEO RECORDERS
# =========================
class CopyingVideoRecorder(VideoRecorder):
    """VideoRecorder robusto che copia i dati prima di processarli."""
    
    def _apply_transform(self, observation):
        if hasattr(observation, 'data'):
            if isinstance(observation.data, numpy.ndarray):
                observation.data = np.array(observation.data).copy()
        return super()._apply_transform(observation)


class SelectiveVideoRecorder(VideoRecorder):
    """Registra solo 1 episodio ogni N episodi."""
    
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
    """Setup dell'ambiente di training."""
    ml1 = metaworld.ML1(cfg.env.name)
    task = ml1.train_tasks[0]
    def make_env():
        gym_env = ml1.train_classes[cfg.env.name](
            render_mode=cfg.env.render_mode,
            camera_name=cfg.env.camera_name
        )
        task = ml1.train_tasks[0]
        gym_env.unwrapped.set_task(task)        
        gym_wrapperd = GymWrapper(gym_env, device=cfg.env_device)
        individual_transforms = [
            StepCounter(max_steps=cfg.env.max_steps),
            FrameSkipTransform(frame_skip=2),
            PixelRenderTransform(out_keys=["pixels"], as_non_tensor=True),
            ToCHWTransform(),
        ]
        return TransformedEnv(gym_wrapperd, Compose(*individual_transforms)).to(cfg.device)
    base_env = ParallelEnv(
        num_workers=cfg.parallel.num_envs,
        create_env_fn=make_env,
        # device=cfg.env_device,
        device='cpu'
    )

    extractor = DinoExtractor(device=cfg.network.device, model_name=cfg.extractor.model_name, output=cfg.extractor.output_type, output_layer=cfg.extractor.output_layer)
    transforms = [
        InitTracker(),
        RewardSum(in_keys=["reward"], out_keys=["reward_sum"]),
        DoubleToFloat(in_keys=["observation"]),
        Resize(w=cfg.extractor.resize_w, h=cfg.extractor.resize_h, in_keys=["pixels"]),
        SelectObservationTransform(),
        UnsqueezeTransform(-4, in_keys=["pixels"]),
        CatFrames(N=cfg.extractor.frame_stack, dim=-4, in_keys=["pixels"]),
        ExtractorTransform(device=cfg.network.device, extractor=extractor, dummy_obs_shape=(1, cfg.extractor.frame_stack, 3, cfg.extractor.resize_h, cfg.extractor.resize_w), out_keys=["pixels_embeddings"]),
    ]
    
    env = TransformedEnv(base_env, Compose(*transforms)).to(cfg.device)
    return env, task, ml1

def setup_test_env(cfg, task, ml1, logger):
    """Setup dell'ambiente di test."""
    gym_env = ml1.train_classes[cfg.env.name](
        render_mode=cfg.env.render_mode,
        camera_name=cfg.env.camera_name
    )
    gym_env.unwrapped.set_task(task)
    extractor = DinoExtractor(device=cfg.network.device, model_name=cfg.extractor.model_name, output=cfg.extractor.output_type, output_layer=cfg.extractor.output_layer)
    transforms = [
        StepCounter(max_steps=cfg.env.max_steps, truncated_key="truncated", step_count_key="step_count"),
        FrameSkipTransform(frame_skip=2),
        DoubleToFloat(in_keys=["observation"]),
        PixelRenderTransform(out_keys=["pixels"], as_non_tensor=True),
        CopyingVideoRecorder(logger=logger, tag="test/video", in_keys=["pixels"]),
        ToCHWTransform(),
        Resize(w=cfg.extractor.resize_w, h=cfg.extractor.resize_h, in_keys=["pixels"]),
        SelectObservationTransform(),
        UnsqueezeTransform(-4, in_keys=["pixels"]),
        CatFrames(N=cfg.extractor.frame_stack, dim=-4, in_keys=["pixels"]),
        UnsqueezeTransform(0, in_keys=["pixels"], allow_positive_dim=True), # Add batch dimension for extractor
        UnsqueezeTransform(0, in_keys=["observation"], allow_positive_dim=True), # Add batch dimension for extractor
        ExtractorTransform(device=cfg.network.device, extractor=extractor, dummy_obs_shape=(1, cfg.extractor.frame_stack, 3, cfg.extractor.resize_h, cfg.extractor.resize_w), out_keys=["pixels_embeddings"]),
    ]
    
    test_env = TransformedEnv(
        GymWrapper(gym_env, device=cfg.device),
        Compose(*transforms))
    # ).to(cfg.device)
    
    return test_env


# =========================
# NETWORK SETUP
# =========================
def setup_networks(cfg, obs_dim, action_dim, action_spec, back_bone_actor=None, back_bone_critic=None):
    """Setup di actor e critic networks."""
    class MultiInputSequential(nn.Module):
        def __init__(self, *modules):
            super().__init__()
            self.modules_list = nn.ModuleList(modules)

        def forward(self, *inputs):
            x = self.modules_list[0](*inputs)
            for module in self.modules_list[1:]:
                x = module(x)
            return x
    
    # Actor
    if back_bone_actor is not None:
        policy_net = MultiInputSequential(
            back_bone_actor,
            MLP(
                in_features=back_bone_actor.output_dim,
                activation_class=nn.ReLU,
                out_features=2 * action_dim,
                num_cells=cfg.sac.actor.hidden_layers
            ),
            NormalParamExtractor()
        )
    else:
        policy_net = nn.Sequential(
            MLP(
                in_features=obs_dim,
                activation_class=nn.ReLU,
                out_features=2 * action_dim,
                num_cells=cfg.sac.actor.hidden_layers
            ),
            NormalParamExtractor()
        )
    policy_td_module = TensorDictModule(
        module=policy_net,
        in_keys=["observation", "pixels_embeddings"],
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
    if back_bone_critic is not None:
        q_net=MultiInputSequential(
            back_bone_critic,
            MLP(
                out_features=1,
                depth=len(cfg.sac.critic.hidden_layers),
                num_cells=cfg.sac.critic.hidden_layers,
            )
        ).to(cfg.device)
    else:
        q_net = MLP(
            in_features=obs_dim + action_dim,
            out_features=1,
            depth=len(cfg.sac.critic.hidden_layers),
            num_cells=cfg.sac.critic.hidden_layers,
        ).to(cfg.device)
    
    qvalue = ValueOperator(
        module=q_net,
        in_keys=["observation", "pixels_embeddings", "action"],
    ).to(cfg.device)
    
    return actor, qvalue


# =========================
# TRAINING HOOKS
# =========================
class VideoLogHook:
    """Hook generico per dump video."""
    
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
    """Logga il reward finale quando l'episodio termina."""
    
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
    """Hook per la valutazione periodica."""
    
    def __init__(self, trainer, test_env, actor, logger, cfg):
        self.trainer = trainer
        self.test_env = test_env
        self.actor = actor
        self.logger = logger
        self.interval = cfg.logging.test_interval
        self.num_episodes = cfg.logging.test_num_episodes
    
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
            if td["action"].ndim == 2 and td["action"].shape[0] == 1:
                td["action"] = td["action"].squeeze(0)
            td = self.test_env.step(td)

            episode_reward += td[("next", "reward")].item()
            terminated = td[("next", "done")].item() if ("next", "done") in td.keys(True) else False
            truncated = td[("next", "truncated")].item()
            done = terminated or truncated
            td = td["next"]
        return episode_reward


def move_batch_to_device(batch, device):
    """Sposta il batch su GPU."""
    return batch.to(device)

import tempfile
import wandb
class CheckpointWandbHook:    
    def __init__(self, trainer, actor, qvalue, logger, cfg):
        self.trainer = trainer
        self.actor = actor
        self.qvalue = qvalue
        self.save_interval = cfg.logging.checkpoint_interval
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


# =========================
# MAIN
# =========================
@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    
    # Setup ambienti
    print("Setting up training environment...")
    train_env, task, ml1 = setup_env(cfg)
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    action_dim = train_env.action_spec.shape[-1]
    
    logger = WandbLogger(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        exp_name=f'{cfg.extractor.model_name}_layer_{cfg.extractor.output_layer}_{cfg.extractor.output_type}_{cfg.env.name}',
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.extractor.model_name, cfg.extractor.output_type, cfg.env.name, f'layer_{cfg.extractor.output_layer}'],
        save_code=True,
    )
    
    test_env = setup_test_env(cfg, task, ml1, logger)
    
    print("Setting up networks...")
    backbone_actor = Model(output_DINO=cfg.extractor.output_type)
    backbone_critic = Model(output_DINO=cfg.extractor.output_type)
    actor, qvalue = setup_networks(cfg, obs_dim, action_dim, train_env.action_spec, back_bone_actor=backbone_actor, back_bone_critic=backbone_critic)
    print("Initializing LazyLinear layers...")
    with torch.no_grad():
        dummy_td = train_env.fake_tensordict()
        _ = actor(dummy_td)
        _ = qvalue(dummy_td)


    print("Setting up collector...")
    collector = Collector(
        train_env,
        actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        init_random_frames=cfg.collector.init_random_frames,
        device=cfg.device,
    )
    
    # Setup replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.replay_buffer.size, device="cpu"),
        batch_size=cfg.replay_buffer.batch_size,
    )
    
    # Setup loss
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        alpha_init=cfg.sac.alpha_init,
        target_entropy=-action_dim,
    )
    loss_module.make_value_estimator(gamma=cfg.sac.gamma)
    
    # Setup optimizer
    optimizer = optim.Adam(loss_module.parameters(), lr=cfg.sac.lr)
    target_net_updater = SoftUpdate(loss_module, tau=cfg.sac.tau)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = SACTrainer(
        collector=collector,
        total_frames=cfg.collector.total_frames,
        replay_buffer=replay_buffer,
        frame_skip=1,
        optim_steps_per_batch=cfg.sac.optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        log_interval=cfg.logging.log_interval,
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


if __name__ == "__main__":
    main()