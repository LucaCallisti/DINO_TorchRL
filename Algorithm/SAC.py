from __future__ import annotations

import warnings
import torch
import torch.cuda
import tqdm
import numpy as np
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, get_available_device, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
import torch.nn as nn
from torchrl.modules.distributions import TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import InteractionType, TensorDictModule




from Algorithm.callbacks import BaseCallback, CallbackList
from torchrl.record import VideoRecorder
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

        self.collector = self._make_collector()
        self.replay_buffer = self._make_replay_buffer()

        self._update_fn = self._build_update_fn()

        self.num_timesteps = 0

    def _build_update_fn(self):
        """Costruisce e avvolge (compile/cudagraph) la funzione di update per efficienza."""
        def update(sampled_tensordict):
            sampled_tensordict = sampled_tensordict.clone().to(self.device)
            loss_td = self.loss_module(sampled_tensordict)
            actor_loss = loss_td["loss_actor"]
            q_loss = loss_td["loss_qvalue"]
            alpha_loss = loss_td["loss_alpha"]

            (actor_loss + q_loss + alpha_loss).sum().backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.target_net_updater.step()
            return loss_td.detach()

        if self.cfg.compile.compile:
            update = compile_with_warmup(update, mode=self.compile_mode, warmup=1)

        if self.cfg.compile.cudagraphs:
            warnings.warn(
                "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                category=UserWarning,
            )
            update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
            
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
        prb = self.cfg.replay_buffer.prb
        eval_iter = self.cfg.logger.eval_iter
        frames_per_batch = self.cfg.collector.frames_per_batch

        collector_iter = iter(self.collector)
        total_iter = len(self.collector)

        self.num_timesteps = 0

        for i in range(total_iter):
            timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

            callback.on_rollout_start()

            # --- COLLECT ---
            with timeit("collect"):
                tensordict = next(collector_iter)

            self.collector.update_policy_weights_()
            
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
                            torch.compiler.cudagraph_mark_step_begin()
                            loss_td = self._update_fn(sampled_tensordict).clone()
                        
                        losses[j] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")

                        if prb:
                            self.replay_buffer.update_priority(sampled_tensordict)

            # --- LOGGING ---
            self._log_and_eval(tensordict, losses if 'losses' in locals() else None, eval_iter, frames_per_batch, pbar)

            # Interrompi se abbiamo raggiunto il numero totale di frame passati a learn()
            if self.num_timesteps >= total_frames:
                break

        callback.on_training_end()
        self._cleanup()

    def _log_and_eval(self, tensordict, losses, eval_iter, frames_per_batch, pbar):
        """Gestisce logging e rollouts di valutazione."""
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["rollout/reward"] = episode_rewards
            metrics_to_log["rollout/episode_length"] = episode_length.sum() / len(episode_length)
        
        if self.num_timesteps >= self.cfg.collector.init_random_frames and losses is not None:
            mean_losses = losses.mean()
            metrics_to_log["train/q_loss"] = mean_losses.get("loss_qvalue")
            metrics_to_log["train/actor_loss"] = mean_losses.get("loss_actor")
            metrics_to_log["train/ent_coef_loss"] = mean_losses.get("loss_alpha")
            metrics_to_log["train/ent_coef"] = self.loss_module.log_alpha.exp().item()
            metrics_to_log["train/learning_rate_actor"] = self.optimizer.param_groups[0]['lr']
            
        # Evaluation
        if abs(self.num_timesteps % eval_iter) < frames_per_batch:
            self.evaluate(num_episodes = self.cfg.logger.num_eval_episodes)
                
        if self.logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/fps"] = pbar.format_dict["rate"]
            self._log_metrics(self.logger, metrics_to_log, self.num_timesteps)

        
    def evaluate(self, num_episodes: int = 1):
        """
        Esegue una valutazione deterministica dell'agente.
        
        :param num_episodes: Numero di episodi su cui calcolare la media.
        :return: Dizionario contenente la ricompensa media e altre metriche.
        """
        avg_reward = 0.0
        # Impostiamo l'esplorazione su DETERMINISTIC per la valutazione
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for _ in range(num_episodes):
                # Eseguiamo un rollout sull'ambiente di valutazione
                eval_rollout = self.eval_env.rollout(
                    max_steps=self.cfg.env.max_episode_steps,
                    policy=self.model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                # Calcoliamo la ricompensa totale dell'episodio
                # reward è solitamente in ("next", "reward")
                avg_reward += eval_rollout["next", "reward"].sum(-2).mean().item()
                
                # Se è prevista la registrazione video, effettuiamo il dump
                self.eval_env.apply(self.dump_video)

        avg_reward /= num_episodes
        
        return {"eval/reward": avg_reward}
    
    def dump_video(self, module):
        if isinstance(module, VideoRecorder):
            module.dump()

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
        loss_module = SACLoss(
            actor_network=self.model[0],
            qvalue_network=self.model[1],
            num_qvalue_nets=2,
            alpha_init=self.cfg.optim.alpha_init
        )
        loss_module.make_value_estimator(gamma=self.cfg.optim.gamma)
        target_net_updater = SoftUpdate(loss_module, eps=self.cfg.optim.target_update_polyak)
        return loss_module, target_net_updater

    def _make_optimizers(self):
        """Inizializza gli ottimizzatori per i vari componenti."""
        from torchrl.objectives import group_optimizers
        opt_actor = optim.Adam(self.model[0].parameters(), lr=self.cfg.optim.lr)
        opt_critic = optim.Adam(self.model[1].parameters(), lr=self.cfg.optim.lr)
        opt_alpha = optim.Adam([self.loss_module.log_alpha], lr=3.0e-4)
        return group_optimizers(opt_actor, opt_critic, opt_alpha)

    def _make_collector(self):
        """Crea il raccoglitore di dati."""
        return SyncDataCollector(
            self.train_env,
            self.exploration_policy,
            frames_per_batch=self.cfg.collector.frames_per_batch,
            total_frames=self.cfg.collector.total_frames,
            device=self.device
            # device='cpu'
        )

    def _make_replay_buffer(self):
        """Configura il buffer di memoria."""
        storage = LazyTensorStorage(max_size=self.cfg.replay_buffer.size, device='cpu') # da provare su cuda
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
    if backbone_actor == None:
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
        in_keys=["observation", "pixels"],
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
    if backbone_critic == None:
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
        in_keys=["observation", "pixels", "action"],
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