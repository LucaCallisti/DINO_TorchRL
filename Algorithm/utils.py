import os
os.environ['MUJOCO_GL']="egl"
os.environ['PYOOPENGL_PLATFORM']="egl"

import functools
from torchrl.envs import (
    GymEnv, TransformedEnv, Compose, ParallelEnv, EnvCreator,
    InitTracker, StepCounter, DoubleToFloat, RewardSum, GymWrapper
)
from torchrl.record import VideoRecorder
from torchrl.record import PixelRenderTransform

# from torchrl.envs.utils import  # set_gym_backend

def env_maker(cfg, device="cpu", from_pixels=False, pre_transforms=None):
    if pre_transforms is None:
        pre_transforms = []

    lib = cfg.env.library
    if lib == "metaworld":
        import metaworld

        ml1 = metaworld.ML1(cfg.env.name)
        if from_pixels:
            env = ml1.train_classes[cfg.env.name](render_mode="rgb_array")
        else:
            env = ml1.train_classes[cfg.env.name]()
        task = ml1.train_tasks[0]
        
        env.set_task(task)
        base_env = GymWrapper(
            env,
            device=device,
        )
         

            
    elif lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")
    if pre_transforms:
        cloned_pre_transforms = [t.clone() for t in pre_transforms]
        return TransformedEnv(base_env, Compose(*cloned_pre_transforms))
    return base_env


def apply_env_transforms(env, max_episode_steps=1000, post_transforms=None):
    if post_transforms is None:
        post_transforms = []

    base_transforms = [
        InitTracker(),
        StepCounter(max_episode_steps),
        DoubleToFloat(),
        RewardSum(),
    ]
    
    all_transforms = base_transforms + post_transforms
    
    transformed_env = TransformedEnv(
        env,
        Compose(*all_transforms),
    )
    return transformed_env


def make_environment(cfg, logger=None, wrapper_pre_parallel_env=None, wrapper_post_parallel_env=None):
    """Make environments for training and evaluation."""
    if wrapper_pre_parallel_env is None: wrapper_pre_parallel_env = []
    if wrapper_post_parallel_env is None: wrapper_post_parallel_env = []

    partial_train_fn = functools.partial(
        env_maker, 
        cfg=cfg, 
        from_pixels=cfg.env.pixel_input,
        pre_transforms=wrapper_pre_parallel_env
    )
    
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial_train_fn),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    # if cfg.env.pixel_input:
    #     parallel_env = TransformedEnv(
    #         parallel_env,
    #         Compose(
    #             PixelRenderTransform(
    #                 out_keys=["pixels"],
    #                 as_non_tensor=True
    #             ),
    #             ToTensorImage(keys=["pixels"]),    # trasforma in [C,H,W]
    #             Resize(224, 224, keys=["pixels"]),
    #             # CatFrames(N=4, keys=["pixels"])  # se vuoi stacking
    #         )
    #     )
    train_env = apply_env_transforms(
        parallel_env, 
        cfg.env.max_episode_steps, 
        post_transforms=wrapper_post_parallel_env
    )

    # partial_eval_fn = functools.partial(
    #     env_maker, 
    #     cfg=cfg, 
    #     from_pixels=cfg.logger.video or cfg.env.pixel_input,
    #     pre_transforms=wrapper_pre_parallel_env
    # )
    
    # trsf_clone = train_env.transform.clone()
    # if cfg.logger.video:
    #     trsf_clone.insert(
    #         0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
    #     )
        
    # eval_env = TransformedEnv(
    #     ParallelEnv(
    #         cfg.collector.env_per_collector,
    #         EnvCreator(partial_eval_fn),
    #         serial_for_single=True,
    #     ),
    #     trsf_clone,
    # )
    eval_env = None
    return train_env, eval_env


def make_train_environment(cfg):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    return train_env