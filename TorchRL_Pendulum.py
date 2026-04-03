import torch
import torch.nn as nn
from torch import optim

from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import Compose, StepCounter, TransformedEnv, RewardSum

from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.modules.models import MLP
from torchrl.modules import ValueOperator

from torchrl.collectors import Collector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
from torchrl.trainers.algorithms import SACTrainer
from tensordict.nn import NormalParamExtractor, TensorDictModule

from torchrl.record import PixelRenderTransform, VideoRecorder

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
# Abilita/disabilita il logging video su WandB
ENABLE_VIDEO = True  # metti False per disattivare

# =========================
# ENV
# =========================
base_env = GymEnv("Pendulum-v1", device=device, render_mode="rgb_array")
transforms = [
    StepCounter(),
    RewardSum(
        in_keys=["reward"], 
        out_keys=["reward_sum"] 
    ),
]

from torchrl.record.loggers import WandbLogger
logger = WandbLogger(project="SAC_Pendulum", entity="CLIP_RL", exp_name="Pendulum")


if ENABLE_VIDEO:
    # Aggiunge il render dei pixel e il recorder video.
    # Il logger viene agganciato più sotto, dopo la sua creazione.
    transforms.append(
        PixelRenderTransform(
            out_keys=["pixels"],
            as_non_tensor=False,
        )
    )
    class SelectiveVideoRecorder(VideoRecorder):
        """Record only 1 episode every N episodes."""
        
        def __init__(self, *args, record_every_n_episodes=5, **kwargs):
            super().__init__(*args, **kwargs)
            self.record_every_n_episodes = record_every_n_episodes
            self.episode_count = 0
            self.should_record = False
        
        def _reset(self, tensordict, tensordict_reset):
            # Increment episode counter on reset
            self.episode_count += 1
            # Record only if episode number is divisible by N
            self.should_record = (self.episode_count % self.record_every_n_episodes == 0)
            return super()._reset(tensordict, tensordict_reset)
        
        def _apply_transform(self, observation):
            # Only apply recording if we should record this episode
            if self.should_record:
                return super()._apply_transform(observation)
            return observation
    transforms.append(
        SelectiveVideoRecorder(
            logger=logger,         # usa il logger creato
            tag="train/video",     # nome della traccia su WandB
            in_keys=["pixels"],
            record_every_n_episodes=5
        )
    )

env = TransformedEnv(
    base_env,
    Compose(*transforms),
).to(device)
obs_dim = env.observation_spec["observation"].shape[-1]
action_dim = env.action_spec.shape[-1]

# =========================
# ACTOR (MLP)
# =========================
policy_net = nn.Sequential(
    MLP(
        in_features=obs_dim, 
        activation_class=nn.ReLU, 
        out_features=2 * action_dim, 
        num_cells=[256, 256] # layer nascosti
    ),
    NormalParamExtractor()
)
policy_td_module = TensorDictModule(
    module=policy_net, 
    in_keys=["observation"], # o il nome della chiave che usi per gli stati
    out_keys=["loc", "scale"]
)

actor = ProbabilisticActor(
    module=policy_td_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low[0],
        "high": env.action_spec.space.high[0],
    },
    return_log_prob=True,
).to(device)

# =========================
# CRITIC (Q network)
# =========================
q_net = MLP(
    in_features=obs_dim + action_dim,
    out_features=1,
    depth=2,
    num_cells=256,
).to(device)

qvalue = ValueOperator(
    module=q_net,
    in_keys=["observation", "action"],
).to(device)

# =========================
# COLLECTOR
# =========================
collector = Collector(
    env,
    actor,
    frames_per_batch=100,
    total_frames=50_000,
    init_random_frames=3000,
    device=device,
)

# =========================
# REPLAY BUFFER
# =========================
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(100_000, device="cpu"),
    batch_size=256,
)


# =========================
# LOSS
# =========================
loss_module = SACLoss(
    actor_network=actor,
    qvalue_network=qvalue,
    num_qvalue_nets=2,
    alpha_init=0.5,
    target_entropy=-action_dim,
)
loss_module.make_value_estimator(gamma=0.99)

# =========================
# OPTIMIZER
# =========================
optimizer = optim.Adam(loss_module.parameters(), lr=3e-4)
target_net_updater = SoftUpdate(loss_module, tau=0.005)

# =========================
# TRAINER
# =========================
trainer = SACTrainer(
    collector=collector,
    total_frames=50_000,
    replay_buffer=replay_buffer,
    frame_skip=1,
    optim_steps_per_batch=10,  
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,  # importante per debug
    log_interval=1000,
    target_net_updater=target_net_updater,
)


## OBBLIGATORIO: sposta il batch su CUDA dopo il sampling, altrimenti il training fallisce perché i tensori sono su CPU.
def move_batch_to_device(batch):
    """Sposta il batch su CUDA dopo il sampling."""
    return batch.to(device)
trainer.register_op("process_optim_batch", move_batch_to_device)


## Per logging video
class VideoDumpHook:
    def __init__(self, trainer, env, interval=10):
        self.trainer = trainer
        self.env = env
        self.interval = interval
        self.step = 0
    
    def __call__(self, batch):
        current_step = self.trainer.collected_frames
        if current_step % self.interval == 0:
            for t in self.env.transform:
                if isinstance(t, VideoRecorder):
                    t.dump(step=current_step)
        return {}

video_hook = VideoDumpHook(trainer, env, interval=1)
trainer.register_op("post_steps_log", video_hook)



# =========================
# TRAIN
# =========================
trainer.train()