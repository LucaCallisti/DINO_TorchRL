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