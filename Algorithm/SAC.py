from __future__ import annotations

import warnings
import torch
import torch.cuda
import tqdm
import numpy as np
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.record import VideoRecorder
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
import torch.nn as nn
from torchrl.modules.distributions import TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import InteractionType, TensorDictModule




from Algorithm.callbacks import BaseCallback, CallbackList
from Algorithm.wrappers import VideoPixelsTransform
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

torch.set_float32_matmul_precision("high")

class SAC:
    def __init__(self, model, exploration_policy, train_env, eval_env, device, cfg):
        self.cfg = cfg
        self.device = device
        
        # Setup Seed
        torch.manual_seed(cfg.optim.seed)
        np.random.seed(cfg.optim.seed)

        # Setup Logger
        self.exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
        self.logger = None
        if cfg.logger.backend:
            self.logger = get_logger(
                logger_type=cfg.logger.backend,
                logger_name="sac_logging",
                experiment_name=self.exp_name,
                wandb_kwargs={
                    "mode": cfg.logger.mode,
                    "entity": cfg.logger.entity,
                    "config": dict(cfg),
                    "project": cfg.logger.project_name,
                    "group": cfg.logger.group_name,
                },
            )

        self.train_env, self.eval_env = train_env, eval_env
        self.model, self.exploration_policy = model, exploration_policy
        self.loss_module, self.target_net_updater = self._make_loss()
        
        self.optimizer = self._make_optimizers()

        self.compile_mode = None
        if cfg.compile.compile:
            self.compile_mode = cfg.compile.compile_mode
            if self.compile_mode in ("", None):
                self.compile_mode = "default" if cfg.compile.cudagraphs else "reduce-overhead"

        self._setup_video_recording()

        self.collector = self._make_collector()
        self.replay_buffer = self._make_replay_buffer()

        self._update_fn = self._build_update_fn()

        self.num_timesteps = 0

    def _setup_video_recording(self):
        """Attach a dedicated video key + VideoRecorder to eval_env."""
        if not self.cfg.logger.video or self.logger is None:
            return

        compose = self.eval_env.transform

        # Insert right after PixelRenderTransform so we preserve raw visual timing.
        insert_pos = None
        for i, t in enumerate(compose):
            if type(t).__name__ == "PixelRenderTransform":
                insert_pos = i + 1
                break
            if type(t).__name__ == "ToCHWTransform" and insert_pos is None:
                # Fallback if PixelRenderTransform isn't visible in the chain.
                insert_pos = i

        if insert_pos is None:
            warnings.warn(
                "VideoRecorder: could not find PixelRenderTransform/ToCHWTransform "
                "in eval_env. Video recording disabled.",
                category=UserWarning,
            )
            return

        self._video_pixels_transform = VideoPixelsTransform(
            in_keys=["pixels"],
            out_keys=["video_pixels"],
        )
        compose.insert(insert_pos, self._video_pixels_transform)

        self._video_recorder = VideoRecorder(
            logger=self.logger,
            tag="eval/video",
            in_keys=["video_pixels"],
        )
        compose.insert(insert_pos + 1, self._video_recorder)

    def _build_update_fn(self):
        """Builds the update function, compiling only the backward+optimizer step.

        NOTE: self.loss_module (SACLoss) is intentionally kept *outside*
        torch.compile because its forward pass uses tensordict.to_module()
        which constructs deeply-nested TensorDict objects that cause
        torch._dynamo to recurse infinitely during tracing.
        """
        def _compiled_backward_and_step(actor_loss, q_loss, alpha_loss):
            """Only this part is compiled — pure tensor ops with no TensorDict nesting.
            zero_grad is called at the START so that gradients from the last backward
            are still readable in _log_and_eval (for grad norm logging).
            """
            self.optimizer.zero_grad(set_to_none=True)   # clear grads BEFORE backward
            (actor_loss + q_loss + alpha_loss).sum().backward()
            self.optimizer.step()
            self.target_net_updater.step()

        if self.cfg.compile.compile:
            _compiled_backward_and_step = compile_with_warmup(
                _compiled_backward_and_step, mode=self.compile_mode, warmup=1
            )

        if self.cfg.compile.cudagraphs:
            warnings.warn(
                "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                category=UserWarning,
            )
            _compiled_backward_and_step = CudaGraphModule(
                _compiled_backward_and_step, in_keys=[], out_keys=[], warmup=5
            )

        def update(sampled_tensordict):
            # loss_module uses tensordict.to_module() internally — must NOT be compiled
            loss_td = self.loss_module(sampled_tensordict)

            actor_loss = loss_td["loss_actor"]
            q_loss    = loss_td["loss_qvalue"]
            alpha_loss = loss_td["loss_alpha"]

            _compiled_backward_and_step(actor_loss, q_loss, alpha_loss)
            return loss_td.detach()

        return update

    def learn(self, total_frames: int = None, callback: BaseCallback | list | None = None):
        """
        Avvia l'addestramento.
        :param total_frames: Sovrascrive i frame totali definiti nel cfg se fornito.
        :param callback: Callback o lista di callback.
        """
        total_frames = total_frames or self.cfg.collector.total_frames
        
        # Gestione Callbacks
        if isinstance(callback, list):
            callback = CallbackList(callback)
        elif callback is None:
            callback = CallbackList([])
        
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        pbar = tqdm.tqdm(total=total_frames)
        
        init_random_frames = self.cfg.collector.init_random_frames
        num_updates = int(self.cfg.collector.frames_per_batch * self.cfg.optim.utd_ratio)
        use_cudagraph_mark = bool(self.cfg.compile.cudagraphs)
        prb = self.cfg.replay_buffer.prb
        eval_iter = self.cfg.logger.eval_iter
        frames_per_batch = self.cfg.collector.frames_per_batch

        collector_iter = iter(self.collector)
        total_iter = len(self.collector)

        self.num_timesteps = 0

        for i in range(total_iter):
            timeit.printevery(num_prints=100_000, total_count=total_iter, erase=True)

            callback.on_rollout_start()

            # --- COLLECT ---
            with timeit("collect"):
                tensordict = next(collector_iter)
            
            current_frames = tensordict.numel()
            self.num_timesteps += current_frames
            pbar.update(current_frames)

            # Trigger Callback On Step
            continue_training = callback.on_step(tensordict)
            if not continue_training:
                print("L'addestramento è stato interrotto dalla callback.")
                break

            with timeit("rb - extend"):
                flat_tensordict = tensordict.reshape(-1)
                self.replay_buffer.extend(flat_tensordict)
                
            callback.on_rollout_end()

            # --- TRAIN ---
            with timeit("train"):
                if self.num_timesteps >= init_random_frames:
                    losses = TensorDict(batch_size=[num_updates])
                    for j in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_tensordict = self.replay_buffer.sample()

                        with timeit("update"):
                            if use_cudagraph_mark:
                                torch.compiler.cudagraph_mark_step_begin()
                            sampled_tensordict = sampled_tensordict.to(self.device)
                            loss_td = self._update_fn(sampled_tensordict)
                        
                        losses[j] = loss_td.select(
                                "loss_actor", "loss_qvalue", "loss_alpha",
                                "entropy",  # needed to decompose actor_loss = -alpha*entropy - Q
                                strict=False,
                            )
                        del loss_td

                        if prb:
                            self.replay_buffer.update_priority(sampled_tensordict)

            # Sync collector policy after gradient updates so next collect uses fresh weights.
            self.collector.update_policy_weights_()

            # --- LOGGING ---
            self._log_and_eval(tensordict, losses if 'losses' in locals() else None, eval_iter, frames_per_batch, pbar)

            # Interrompi se abbiamo raggiunto il numero totale di frame passati a learn()
            if self.num_timesteps >= total_frames:
                break

        callback.on_training_end()
        self._cleanup()

    def _log_and_eval(self, tensordict, losses, eval_iter, frames_per_batch, pbar):
        """Handles logging and evaluation rollouts."""
        episode_end = tensordict["next", "done"] | tensordict["next", "truncated"]
        episode_rewards = tensordict["next", "episode_reward"][episode_end]
        step_rewards = tensordict["next", "reward"]
        actions = tensordict.get("action", None)

        metrics_to_log = {}
        metrics_to_log["progress/num_timesteps"] = self.num_timesteps
        metrics_to_log["rollout/reward_step_mean"] = step_rewards.float().mean().item()
        metrics_to_log["rollout/reward_step_std"] = step_rewards.float().std().item()
        if actions is not None:
            actions_float = actions.float()
            action_std = actions_float.std()
            action_mean_abs = actions_float.abs().mean()
            if torch.isfinite(action_std):
                metrics_to_log["rollout/action_std"] = action_std.item()
            if torch.isfinite(action_mean_abs):
                metrics_to_log["rollout/action_mean_abs"] = action_mean_abs.item()
        if episode_rewards.numel() > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["rollout/reward"] = episode_rewards.mean().item()
            metrics_to_log["rollout/reward_std"] = episode_rewards.float().std().item()
            metrics_to_log["rollout/episode_length"] = episode_length.float().mean().item()
            metrics_to_log["rollout/episodes_in_batch"] = int(episode_end.sum().item())
        
        if self.num_timesteps >= self.cfg.collector.init_random_frames and losses is not None:
            mean_losses = losses.mean()
            actor_loss = mean_losses.get("loss_actor")
            entropy    = mean_losses.get("entropy", None)
            alpha      = self.loss_module.log_alpha.exp().detach()

            metrics_to_log["train/q_loss"]           = mean_losses.get("loss_qvalue")
            metrics_to_log["train/actor_loss"]        = actor_loss
            metrics_to_log["train/ent_coef_loss"]     = mean_losses.get("loss_alpha")
            metrics_to_log["train/ent_coef"]          = alpha.item()
            metrics_to_log["train/learning_rate_actor"] = self.optimizer.param_groups[0]['lr']

            # Decompose actor_loss = -alpha*entropy - mean_Q  →  mean_Q = -(actor_loss + alpha*entropy)
            # Useful to check whether Q is growing even when actor_loss appears to increase
            if entropy is not None:
                metrics_to_log["train/entropy"]      = entropy.item()
                mean_q_actor = -(actor_loss + alpha * entropy)
                metrics_to_log["train/mean_q_actor"] = mean_q_actor.item()

            # Gradient norms — useful to detect vanishing/exploding gradients in backbone
            def _grad_norm(module):
                total = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        total += p.grad.detach().norm().item() ** 2
                return total ** 0.5

            if self.cfg.logger.grad_norm:
                actor_mod = getattr(self.loss_module, "actor_network", self.model[0])
                critic_mod = getattr(self.loss_module, "qvalue_network", self.model[1])
                metrics_to_log["train/grad_norm_actor"]  = _grad_norm(actor_mod)
                metrics_to_log["train/grad_norm_critic"] = _grad_norm(critic_mod)

        # Evaluation
        if self.eval_env is not None and abs(self.num_timesteps % eval_iter) < frames_per_batch:
            eval_metrics = self.evaluate(num_episodes=self.cfg.logger.num_eval_episodes)
            metrics_to_log.update(eval_metrics)

        if self.logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/fps"] = pbar.format_dict["rate"]
            self._log_metrics(self.logger, metrics_to_log, self.num_timesteps)


    def evaluate(self, num_episodes: int = 1):
        """
        Runs a deterministic evaluation of the agent.

        :param num_episodes: Number of episodes to average over.
        :return: Dict with eval metrics.
        """
        avg_reward = 0.0
        record_video = hasattr(self, "_video_recorder") and hasattr(self, "_video_pixels_transform")
        if record_video:
            self._video_recorder.obs = []
            self._video_recorder.count = 0
            self._video_recorder.skip = 1

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for ep in range(num_episodes):
                if record_video:
                    # Record only first eval episode to keep videos short and consistent.
                    if ep == 1:
                        self._video_recorder.skip = 10**9
                        self._video_recorder.count = 0
                eval_rollout = self.eval_env.rollout(
                    max_steps=self.cfg.env.max_episode_steps,
                    policy=self.model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
                avg_reward += eval_rollout["next", "reward"].sum(-2).mean().item()

        avg_reward /= num_episodes

        if record_video:
            self._video_recorder.dump(step=self.num_timesteps)
            self._video_recorder.skip = 1

        return {"eval/reward": avg_reward}

    def _cleanup(self):
        self.collector.shutdown()
        if not self.eval_env.is_closed:
            self.eval_env.close()
        if not self.train_env.is_closed:
            self.train_env.close()

    def save(self, path: str):
        """Esempio di funzione di salvataggio del modello."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def _make_loss(self):
        """Configura il modulo della loss SAC."""
        action_spec = self.train_env.action_spec
        if hasattr(action_spec, "keys") and "action" in action_spec.keys():
            action_shape = action_spec["action"].shape
        else:
            action_shape = action_spec.shape
        action_dim = action_shape[-1]

        loss_module = SACLoss(
            actor_network=self.model[0],
            qvalue_network=self.model[1],
            num_qvalue_nets=2,
            alpha_init=self.cfg.optim.alpha_init,
            target_entropy = -self.cfg.optim.alpha_target * action_dim
        )
        loss_module.make_value_estimator(gamma=self.cfg.optim.gamma)
        target_net_updater = SoftUpdate(loss_module, eps=self.cfg.optim.target_update_polyak)
        return loss_module, target_net_updater

    def _make_optimizers(self):
        """Inizializza gli ottimizzatori per i vari componenti."""
        from torchrl.objectives import group_optimizers
        opt_actor = optim.Adam(self.model[0].parameters(), lr=self.cfg.optim.lr, capturable=True)
        opt_critic = optim.Adam(self.model[1].parameters(), lr=self.cfg.optim.lr, capturable=True)
        opt_alpha = optim.Adam([self.loss_module.log_alpha], lr=self.cfg.optim.alpha_lr, capturable=True)
        return group_optimizers(opt_actor, opt_critic, opt_alpha)

    def _make_collector(self):
        """Crea il raccoglitore di dati."""
        return SyncDataCollector(
            self.train_env,
            self.exploration_policy,
            frames_per_batch=self.cfg.collector.frames_per_batch,
            total_frames=self.cfg.collector.total_frames,
            device=self.device
        )

    def _make_replay_buffer(self):
        """Configura il buffer di memoria."""
        storage = LazyTensorStorage(max_size=self.cfg.replay_buffer.size, device=self.cfg.replay_buffer.device) # da provare su cuda
        return TensorDictReplayBuffer(storage=storage, batch_size=self.cfg.optim.batch_size)

    def _log_metrics(self, logger, metrics, step):
        """Iterates through a dictionary of metrics and logs them individually."""
        for key, value in metrics.items():
            # Verifica che il valore sia un numero o un tensore scalare
            if isinstance(value, (int, float, torch.Tensor)):
                logger.log_scalar(key, value, step=step)


def make_sac_agent(cfg, train_env, backbone_actor=None, backbone_critic=None, device = 'cpu'):
    """Make SAC agent."""
    # Define Actor Network
    action_spec = train_env.action_spec_unbatched.to(device)
    if backbone_actor is None:
        actor_net = MLP(
            num_cells=cfg.network.hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=get_activation(cfg),
            device=device,
        )
    else:
        actor_head = MLP(
            num_cells=[],
            out_features=2 * action_spec.shape[-1],
            activation_class=get_activation(cfg),
            device=device,
        )
        actor_net = MultiInputSequential(backbone_actor.to(device), actor_head)

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    ).to(device)
    actor_net = MultiInputSequential(actor_net, actor_extractor)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation", "pixels_embeddings"],
        out_keys=["loc", "scale"]
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
    if backbone_critic is None:
        qvalue_net = MLP(
            num_cells=cfg.network.hidden_sizes,
            out_features=1,
            activation_class=get_activation(cfg),
            device=device,
        )
    else:
        qvalue_net_head = MLP(
            num_cells=[],
            out_features=1,
            activation_class=get_activation(cfg),
            device=device,
        )
        qvalue_net = MultiInputSequential(backbone_critic.to(device), qvalue_net_head)

    qvalue = ValueOperator(
        in_keys=["observation", "pixels_embeddings", "action"],
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue])

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = train_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)

    return model, model[0]


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


class MultiInputSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *inputs):
        x = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            x = module(x)
        return x