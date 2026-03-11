import functools
from torchrl.envs import ParallelEnv, TransformedEnv, EnvCreator
from torchrl.record import VideoRecorder
import hydra
from omegaconf import DictConfig
# Importa qui anche la tua classe SAC e le utility
from Algorithm.SAC import SAC
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
import torch.nn as nn
from tensordict.nn import TensorDictModule, InteractionType
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.distributions import TanhNormal
from torchrl.envs.utils import ExplorationType, set_exploration_type
import torch
from torchrl.envs.transforms import Compose, InitTracker, StepCounter, DoubleToFloat, RewardSum


def make_environment(device):
    """Make environments for training and evaluation."""
    
    train_env = GymEnv(
        'HalfCheetah-v4',
        device=device,
        from_pixels=False,
        pixels_only=False,
    )
    eval_env = GymEnv(
        'HalfCheetah-v4',
        device=device,
        from_pixels=False,
        pixels_only=False,
    )
    return train_env, eval_env

def make_sac_agent(train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec_unbatched.to(device)

    actor_net = MLP(
        num_cells=[256, 256],
        out_features=2 * action_spec.shape[-1],
        activation_class=nn.ReLU,
        device=device,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{1}",
        scale_lb=0.1,
    ).to(device)
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net = MLP(
        num_cells=[256, 256],
        out_features=1,
        activation_class=nn.ReLU,
        device=device,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue])

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    return model, model[0]


@hydra.main(version_base="1.1", config_path="Algorithm", config_name="SAC_config")
def main(cfg):
    # 1. Prepariamo gli ambienti usando le utility
    train_env, eval_env = make_environment(device="cuda:0")

    train_env = TransformedEnv(
        train_env,
        Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),  
            RewardSum(),
        ),
    )
    eval_env = TransformedEnv(
        eval_env,
        Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),  
            RewardSum(),
        ),
    )
    
    # 2. Creiamo l'agente (modelli actor e qvalue)
    # Qui potresti chiamare una tua funzione custom o quella in utils.py
    model, actor_exploration = make_sac_agent(train_env, eval_env, device="cuda:0")

    # 3. Inizializziamo il Trainer SAC con i pezzi pronti
    trainer = SAC(
        cfg=cfg,
        model=model,
        exploration_policy=actor_exploration,
        train_env=train_env,
        eval_env=eval_env,
        device="cuda:0"
    )
    
    # 4. Via all'addestramento!
    trainer.learn()

if __name__ == "__main__":
    main()
