import torch
import numpy as np
from torchrl.envs import Transform
from torchrl.data.tensor_specs import Bounded, UnboundedContinuous
from tensordict import TensorDictBase
import gymnasium as gym

class ToCHWTransform(Transform):
    def __init__(self, in_keys=None, out_keys=None, frame_shape=(3, 480, 480)):
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = ["pixels"]
        self.frame_shape = frame_shape
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase = None) -> TensorDictBase:
        if tensordict_out is None:
            tensordict_out = tensordict_in
        pixels = tensordict_out.get(self.in_keys[0])
        normalized_pixels = self._process(pixels)
        tensordict_out.set( self.out_keys[0], normalized_pixels)
        return tensordict_out

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


class ObservationSliceTransform(Transform):
    def __init__(self, in_keys=None, out_keys=None, slice_len=4):
        if in_keys is None:
            in_keys = ["observation"]  # chiave di default
        if out_keys is None:
            out_keys = ["observation"]  # sovrascrive le osservazioni originali
            
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.slice_len = slice_len

    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase = None) -> TensorDictBase:
        obs = tensordict_out.get(self.in_keys[0])
        tensordict_out.set(self.out_keys[0], obs[..., :self.slice_len])
        return tensordict_out

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        obs = tensordict_reset.get(self.in_keys[0])
        tensordict_reset.set(self.out_keys[0], obs[..., :self.slice_len])
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        old_spec = observation_spec[self.out_keys[0]]
        new_shape = observation_spec.shape + (self.slice_len,)
        new_spec = UnboundedContinuous(
            shape=new_shape,
            dtype=old_spec.dtype,
            device=old_spec.device
        )
        observation_spec.set(self.out_keys[0], new_spec)
        return observation_spec


class RunningRewardNorm(Transform):
    """
    Normalizes rewards using Welford's online algorithm (running mean + variance).
    Device-agnostic: automatically moves internal stats to match the reward tensor's device.
    Equivalent to SB3's VecNormalize for rewards, without VecNormV2's batched-env issues.

    Args:
        in_keys:  source key(s), default ["reward"]
        out_keys: destination key(s), default same as in_keys (overwrites in-place).
                  Pass e.g. out_keys=["reward_normalized"] to keep the raw reward intact.
        eps:      epsilon added to std to avoid division by zero
        clip:     clips normalized reward to [-clip, clip]
    """
    def __init__(self, in_keys=None, out_keys=None, eps: float = 1e-8, clip: float = 10.0):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = in_keys  # overwrite in-place by default
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.eps = eps
        self.clip = clip
        # Welford state — kept as plain tensors, moved to device lazily
        self.register_buffer("_count", torch.zeros(1))
        self.register_buffer("_mean",  torch.zeros(1))
        self.register_buffer("_M2",    torch.zeros(1))  # sum of squared deviations

    def _update_stats(self, x: torch.Tensor):
        """Welford online update (batched)."""
        # Move internal state to same device as data on first call / if device changes
        if self._count.device != x.device:
            self._count = self._count.to(x.device)
            self._mean  = self._mean.to(x.device)
            self._M2    = self._M2.to(x.device)

        x_flat = x.flatten().float()
        for val in x_flat:
            self._count += 1
            delta  = val - self._mean
            self._mean += delta / self._count
            delta2 = val - self._mean
            self._M2   += delta * delta2

    @property
    def _std(self) -> torch.Tensor:
        variance = self._M2 / self._count.clamp(min=1)
        return variance.sqrt().clamp(min=self.eps)

    def _normalize(self, reward: torch.Tensor) -> torch.Tensor:
        self._update_stats(reward)
        normed = (reward - self._mean) / self._std
        return normed.clamp(-self.clip, self.clip)

    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase) -> TensorDictBase:
        reward = tensordict_out.get(self.in_keys[0])
        tensordict_out.set(self.out_keys[0], self._normalize(reward))
        return tensordict_out

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # Rewards are not present at reset — nothing to do
        return tensordict_reset


class VideoPixelsTransform(Transform):
    """Creates a dedicated key for video logging (e.g. video_pixels).

    This lets us keep the training key (pixels) untouched while feeding
    TorchRL VideoRecorder with the exact format we want.
    """

    def __init__(self, in_keys=None, out_keys=None):
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = ["video_pixels"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _to_video_tensor(self, frame):
        if type(frame).__name__ in ["NonTensorStack", "NonTensorData"]:
            frame = torch.from_numpy(np.asarray(frame.tolist()))
        elif not isinstance(frame, torch.Tensor):
            frame = torch.as_tensor(frame)

        # If CHW, convert to HWC for compatibility with recorder expectations.
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)
        elif frame.ndim == 4 and frame.shape[1] in (1, 3):
            frame = frame.permute(0, 2, 3, 1)

        if frame.dtype.is_floating_point:
            if frame.max() <= 1.0:
                frame = (frame * 255.0).clamp(0, 255)
            frame = frame.to(torch.uint8)
        else:
            frame = frame.to(torch.uint8)
        return frame

    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase) -> TensorDictBase:
        frame = tensordict_out.get(self.in_keys[0], None)
        if frame is not None:
            tensordict_out.set(self.out_keys[0], self._to_video_tensor(frame))
        return tensordict_out

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        frame = tensordict_reset.get(self.in_keys[0], None)
        if frame is not None:
            tensordict_reset.set(self.out_keys[0], self._to_video_tensor(frame))
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        # Register the additional key so rollout/spec building remains consistent.
        old_spec = observation_spec[self.in_keys[0]]
        observation_spec[self.out_keys[0]] = old_spec.clone()
        return observation_spec


class EvalFrameCapture(Transform):
    """
    Passively captures the first-environment's pixel frame at each env step.
    Does NOT log to WandB itself — frames are accessed and logged manually at
    the correct training step via SAC.evaluate(), avoiding VideoRecorder's
    internal step-counter conflict with WandB's global step.

    Must be placed after ToCHWTransform so frames are real CHW float32 images,
    not DINO embeddings.
    """
    def __init__(self, in_keys=None):
        if in_keys is None:
            in_keys = ["pixels"]
        super().__init__(in_keys=in_keys, out_keys=in_keys)
        self.frames: list = []
        self.capture_enabled: bool = True

    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase) -> TensorDictBase:
        if not self.capture_enabled:
            return tensordict_out
        frame = tensordict_out.get(self.in_keys[0])  # (B, C, H, W) float32
        if frame is not None and frame.dim() >= 3:
            self.frames.append(frame[0].detach().cpu())  # capture env[0] only
        return tensordict_out

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return tensordict_reset

    def clear(self):
        self.frames.clear()

    def set_enabled(self, enabled: bool):
        self.capture_enabled = enabled

    def get_video_tensor(self):
        """Returns (T, C, H, W) uint8 tensor ready for wandb.Video, or None if empty."""
        if not self.frames:
            return None
        video = torch.stack(self.frames)           # (T, C, H, W) float32 [0-1]
        video = (video * 255).clamp(0, 255).byte() # (T, C, H, W) uint8
        return video