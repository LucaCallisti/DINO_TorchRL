# callbacks.py

class BaseCallback:
    """
    Classe base per le callback in stile Stable Baselines3, 
    adattata per funzionare con i TensorDict di TorchRL.
    """
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.n_calls = 0
        self.num_timesteps = 0
        self.model = None  # Verrà iniettato dalla classe SAC

    def init_callback(self, model) -> None:
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        self._on_training_start(locals_, globals_)

    def _on_training_start(self, locals_: dict, globals_: dict) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def on_step(self, tensordict) -> bool:
        """
        Chiamata ad ogni iterazione del collector.
        :param tensordict: Il TensorDict raccolto nell'ultimo step/batch.
        :return: Se False, interrompe l'addestramento in anticipo.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step(tensordict)

    def _on_step(self, tensordict) -> bool:
        return True

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass


class CallbackList(BaseCallback):
    """Gestisce una lista di callback."""
    def __init__(self, callbacks: list):
        super().__init__()
        self.callbacks = [c for c in callbacks if c is not None]

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self, locals_: dict, globals_: dict) -> None:
        for callback in self.callbacks:
            callback.on_training_start(locals_, globals_)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self, tensordict) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = continue_training and callback.on_step(tensordict)
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()


class EmbeddingDiversityCallback(BaseCallback):
    """
    Logs DINO embedding diversity metrics to detect feature collapse.

    Samples a mini-batch from the replay buffer every `log_every` steps and computes:
    - embed/mean_cos_sim  : mean pairwise cosine similarity.
                            Close to 1.0 → all states look the same (collapse).
                            Ideally < 0.5 for a diverse feature space.
    - embed/mean_feat_std : mean per-dimension std across the batch.
                            Low → embeddings are clustered in a small region.
    - embed/mean_norm     : mean L2 norm — sanity-check for embedding scale.

    Args:
        log_every: how many training steps between each logging call (default 1000).
        key:       tensordict key holding the embeddings (default "pixels_embeddings").
    """
    def __init__(self, log_every: int = 1_000, key: str = "pixels_embeddings", verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_every = log_every
        self.key = key
        self._last_log_step = -log_every  # Force log on first call

    def _on_step(self, tensordict) -> bool:
        if self.num_timesteps - self._last_log_step < self.log_every:
            return True  # not yet time to log
        
        self._last_log_step = self.num_timesteps

        # Replay buffer must have data
        rb = self.model.replay_buffer
        if len(rb) == 0:
            return True

        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            batch = rb.sample()
            feats = batch.get(self.key, None)
            if feats is None and isinstance(self.key, str):
                feats = batch.get(("next", self.key), None)

        if feats is None:
            return True

        feats = feats.float()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        elif feats.dim() > 2:
            feats = feats.flatten(start_dim=1)

        B = feats.shape[0]

        fn = F.normalize(feats, dim=-1)            # (B, D)
        cos_sim = fn @ fn.T                         # (B, B)
        off_diag_mean = (cos_sim.sum() - cos_sim.trace()) / max(B * (B - 1), 1)

        metrics = {
            "embed/mean_cos_sim":  off_diag_mean.item(),
            "embed/mean_feat_std": feats.std(dim=0).mean().item(),
            "embed/mean_norm":     feats.norm(dim=-1).mean().item(),
        }

        if self.model.logger is not None:
            self.model._log_metrics(self.model.logger, metrics, self.num_timesteps)

        return True