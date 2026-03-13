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

from torchrl.envs.transforms import FrameSkipTransform, CatFrames, ToTensorImage, Resize, UnsqueezeTransform
from torchrl.record import PixelRenderTransform
from Model.extractor import ExtractorTransform, DinoExtractor, Model
import copy

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase
import numpy as np
from Algorithm.wrappers import ToCHWTransform, ObservationSliceTransform




@hydra.main(version_base="1.1", config_path="Algorithm", config_name="SAC_config")
def main(cfg):
    # 1. Prepariamo gli ambienti usando le utility
    extractor = DinoExtractor(device=cfg.network.device, model_name='vits16_ft', output='Attention_Pooling')

    wrapper_pre_parallel_env=[FrameSkipTransform(frame_skip=2)]
    wrapper_post_parallel_env=[
        PixelRenderTransform(
                    out_keys=["pixels"],
                    as_non_tensor=False
                ),
        ToCHWTransform(),
        Resize(w=224, h=224, in_keys=["pixels"]),
        ObservationSliceTransform(slice_len=4),
        UnsqueezeTransform(-4, in_keys=["pixels"]),
        CatFrames(N=4, dim=-4, in_keys=["pixels"]),
        CatFrames(N=4, dim=-1, in_keys=["observation"]),
        ExtractorTransform(device=cfg.network.device, extractor=extractor),
        ]
    train_env, eval_env = make_environment(cfg, wrapper_pre_parallel_env=wrapper_pre_parallel_env, wrapper_post_parallel_env=wrapper_post_parallel_env)
    backbone_actor = Model(n_frame=4, device=cfg.network.device)
    model, actor_exploration = make_sac_agent(cfg, train_env, backbone_actor=backbone_actor, backbone_critic=copy.deepcopy(backbone_actor), device="cuda:0")

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
