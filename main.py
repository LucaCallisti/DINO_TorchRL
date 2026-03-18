import hydra
# Importa qui anche la tua classe SAC e le utility

from Algorithm.SAC import SAC, make_sac_agent
from Algorithm.utils import make_environment

from torchrl.envs.transforms import FrameSkipTransform, CatFrames, Resize, UnsqueezeTransform
from torchrl.record import PixelRenderTransform
from Model.extractor import ExtractorTransform, DinoExtractor, Model
import copy

from Algorithm.wrappers import ToCHWTransform, ObservationSliceTransform, RunningRewardNorm
from Algorithm.callbacks import EmbeddingDiversityCallback




@hydra.main(version_base="1.1", config_path="Algorithm", config_name="SAC_config")
def main(cfg):
    # 1. Prepariamo gli ambienti usando le utility
    extractor = DinoExtractor(device=cfg.network.device, model_name='vits16_ft', output='Attention_Pooling')

    wrapper_pre_parallel_env=[FrameSkipTransform(frame_skip=2)]
    wrapper_post_parallel_env=[
        RunningRewardNorm(in_keys=["reward"]),
        PixelRenderTransform(
                    out_keys=["pixels"],
                    as_non_tensor=False,
                ),
        ToCHWTransform(),
        Resize(w=224, h=224, in_keys=["pixels"]),
        ObservationSliceTransform(slice_len=4),
        UnsqueezeTransform(-4, in_keys=["pixels"]),
        CatFrames(N=4, dim=-4, in_keys=["pixels"]),
        CatFrames(N=4, dim=-1, in_keys=["observation"]),
        ExtractorTransform(device=cfg.network.device, extractor=extractor, out_keys=["pixels_embeddings"]),
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


    td = train_env.reset()
    px = td.get("pixels", None)   

    from pathlib import Path
    from PIL import Image
    import torch
    if px is not None:
        # Riduce eventuali dimensioni batch/tempo
        while px.dim() > 3:
            px = px[0]
        # CHW -> HWC
        if px.dim() == 3 and px.shape[0] in (1, 3):
            px = px.permute(1, 2, 0)
        # float [0,1] -> uint8
        if px.dtype.is_floating_point:
            px = (px.clamp(0, 1) * 255).to(torch.uint8)
        else:
            px = px.to(torch.uint8)

        out = Path("/home/l.callisti/DINO_Torch_RL/Code/pretrain_frame.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(px.cpu().numpy()).save(out)
    

    # 4. Via all'addestramento!
    trainer.learn(callback=EmbeddingDiversityCallback(log_every=5_000))

if __name__ == "__main__":
    main()
