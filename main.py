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
from Model.extractor import ExtractorTransform, DinoExtractor

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase
import numpy as np
from torchrl.data.tensor_specs import Bounded
class Custom(Transform):
    def __init__(self, in_keys=None, out_keys=None, frame_shape=(3, 480, 480)):
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = ["pixels"]
        self.frame_shape = frame_shape
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        pixels = tensordict.get(("next", self.in_keys[0]))
        normalized_pixels = self._process(pixels)
        tensordict.set(("next", self.out_keys[0]), normalized_pixels)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        pixels = tensordict_reset.get(self.in_keys[0])
        normalized_pixels = self._process(pixels)
        tensordict_reset.set(self.out_keys[0], normalized_pixels)
        return tensordict_reset
    
    def _process(self, frame):
        if type(frame).__name__ in ["NonTensorStack", "NonTensorData"]:
            raw_numpy_list = frame.tolist()
            frame = torch.stack([torch.from_numpy(arr) for arr in raw_numpy_list])
        elif not isinstance(frame, torch.Tensor):
            frame = torch.as_tensor(frame)
        frame = frame.to(torch.float32).div_(255.0)
        if frame.ndim == 4:
            frame = frame.permute(0, 3, 1, 2)
        elif frame.ndim == 3:
            frame = frame.permute(2, 0, 1)
        return frame
    
    def transform_observation_spec(self, observation_spec):
        old_spec = observation_spec[self.out_keys[0]]
        shape = torch.Size(observation_spec.shape + self.frame_shape)
        low_tensor = torch.zeros(shape, device=old_spec.device, dtype=torch.float32)
        high_tensor = torch.ones(shape, device=old_spec.device, dtype=torch.float32)
        new_spec = Bounded(
            low=low_tensor,
            high=high_tensor,
            shape=shape,
            dtype=torch.float32,
            device=old_spec.device
        )
        observation_spec[self.out_keys[0]] = new_spec
        return observation_spec

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
        Custom(),
        Resize(w=224, h=224, in_keys=["pixels"]),
        UnsqueezeTransform(-4, in_keys=["pixels"]),
        CatFrames(N=4, dim=-4, in_keys=["pixels"]),
        ExtractorTransform(device=cfg.network.device, extractor=extractor),
        ]
    train_env, eval_env = make_environment(cfg, wrapper_pre_parallel_env=wrapper_pre_parallel_env, wrapper_post_parallel_env=wrapper_post_parallel_env)
    td = train_env.reset()
    breakpoint()
    
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
