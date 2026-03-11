import functools
from torchrl.envs import (
    GymEnv, TransformedEnv, Compose, ParallelEnv, EnvCreator,
    InitTracker, StepCounter, DoubleToFloat, RewardSum
)
from torchrl.record import VideoRecorder
from torchrl.envs.utils import set_gym_backend

def env_maker(cfg, device="cpu", from_pixels=False, pre_transforms=None):
    if pre_transforms is None:
        pre_transforms = []

    lib = cfg.env.library
    if lib == "metaworld":
        import metaworld
        ml1 = metaworld.ML1(cfg.env.name) 
        raw_env = ml1.train_classes[cfg.env.name]()
        task = ml1.train_tasks[0]
        raw_env.set_task(task)
        base_env = GymWrapper(raw_env, device=device)
    elif lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )

            if pre_transforms:
                return TransformedEnv(base_env, Compose(*pre_transforms))
            return base_env
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


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

    train_env = apply_env_transforms(
        parallel_env, 
        cfg.env.max_episode_steps, 
        post_transforms=wrapper_post_parallel_env
    )

    partial_eval_fn = functools.partial(
        env_maker, 
        cfg=cfg, 
        from_pixels=cfg.logger.video or cfg.env.pixel_input,
        pre_transforms=wrapper_pre_parallel_env
    )
    
    trsf_clone = train_env.transform.clone()
    if cfg.logger.video:
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
        
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(partial_eval_fn),
            serial_for_single=True,
        ),
        trsf_clone,
    )
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