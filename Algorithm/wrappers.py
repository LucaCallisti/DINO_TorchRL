import torch
from torchrl.envs import Transform
from torchrl.data.tensor_specs import Bounded, UnboundedContinuous
from tensordict import TensorDictBase

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