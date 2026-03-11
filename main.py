import functools
from torchrl.envs import ParallelEnv, TransformedEnv, EnvCreator
from torchrl.record import VideoRecorder
import hydra
from omegaconf import DictConfig
# Importa qui anche la tua classe SAC e le utility
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
import torch.nn as nn
from tensordict.nn import TensorDictModule, InteractionType
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.distributions import TanhNormal
from torchrl.envs.utils import ExplorationType, set_exploration_type
import torch
from torchrl.envs.transforms import Compose, InitTracker, StepCounter, DoubleToFloat, RewardSum

from Algorithm.SAC import SAC, make_sac_agent
from Algorithm.utils import make_environment


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
