import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torchrl.envs.transforms import Transform
from pathlib import Path

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import UnboundedContinuous

class ExtractorTransform(Transform):
    def __init__(self, device, extractor, dummy_obs_shape, in_keys=None, out_keys=None):
        if in_keys is None:
            in_keys = ["pixels"]
        if out_keys is None:
            out_keys = ["pixels"] # Sovrascrive i pixel con gli embedding
            
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        
        self.extractor = extractor
        self.extractor.to(device)
        self.extractor.eval()

        dummy_obs = torch.zeros(dummy_obs_shape, device=extractor.device)
        with torch.no_grad():
            dummy_output = self.extractor(dummy_obs)
            self.embedding_shape = dummy_output.shape[1:]
    
    def _step(self, tensordict_in: TensorDictBase, tensordict_out: TensorDictBase = None) -> TensorDictBase:
        pixels = tensordict_out.get(self.in_keys[0])
        normalized_pixels = self._process(pixels)
        tensordict_out.set(self.out_keys[0], normalized_pixels)
        return tensordict_out

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        pixels = tensordict_reset.get(self.in_keys[0])
        normalized_pixels = self._process(pixels)
        tensordict_reset.set(self.out_keys[0], normalized_pixels)
        return tensordict_reset

    def _process(self, obs):
        with torch.no_grad():
            obs_gpu = obs.to(self.extractor.device)
            embeddings = self.extractor(obs_gpu)
        return embeddings

    def transform_observation_spec(self, observation_spec):
        old_spec = observation_spec[self.in_keys[0]]
        batch_shape = observation_spec.shape

        new_shape = batch_shape + self.embedding_shape
        new_spec = UnboundedContinuous(
            shape=new_shape,
            dtype=torch.float32,
            device=old_spec.device
        )
        observation_spec.set(self.out_keys[0], new_spec)        
        return observation_spec

    def to(self, device):
        super().to(device)
        return self
    
    
    
class DinoExtractor(nn.Module):
    MODELS = {
        "vits16_ft": ("dinov3_vits16",  "/home/l.callisti/DINO_Torch_RL/Code/DINO/dino_finetuned_multicrop_200e.pth",  384),
        "vits16":  ("dinov3_vits16",  "/home/l.callisti/DINO_Torch_RL/Code/DINO/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",  384),
        "vitb16":  ("dinov3_vitb16",  "/home/l.callisti/DINO_Torch_RL/Code/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",  768),
    }

    def __init__(self, device: torch.device, model_name: str = "vitb16", output : str = 'cls'):
        super().__init__()
        self.device = device
        self.output = output

        hub_name, weights_path, self.embed_dim = self.MODELS[model_name]
        current_file_path = Path(__file__).resolve()
        
        self.model = torch.hub.load(
            '/home/l.callisti/DINO_Torch_RL/Code/dinov3',
            hub_name,
            source="local",
            weights=weights_path
        ).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = torch.compile(self.model).to(device)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
        
    def forward(self, img):
        '''
        img: (B, N_frames, C, H, W)
        '''
        with torch.no_grad():
            init_shape = img.shape # (B, N_frames, C, H, W)
            image = img.reshape(-1, init_shape[-3], init_shape[-2], init_shape[-1])
            image = (image - self.mean) / self.std
            features = self.model.forward_features(image, masks=None)
            if self.output == 'cls':
                output = features['x_norm_clstoken']
                output = output.reshape(init_shape[0], init_shape[1], -1)
                output = output.reshape(init_shape[0], -1)
            else:
                output = features['x_norm_patchtokens']
                output = output.reshape(init_shape[0], init_shape[1], output.shape[-2], output.shape[-1])
        return output


class Attention_Pooling(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm_in = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_frames, N_patch, D = x.shape
        # x_flat = x.reshape(B, N_frames * N_patch, D).contiguous()   # messo i N_frame in N_patch e non in batch
        x_flat = x.contiguous().reshape(B * N_frames, N_patch, D)

        # x_flat = self.norm_in(x_flat)
        queries = self.query_token.repeat(B * N_frames, 1, 1)
        pooled_frames = self.cross_attention(queries, x_flat, x_flat)[0]
        pooled_frames = self.norm_out(pooled_frames + queries)
        sequence_of_frames = pooled_frames.reshape(B, N_frames, D)        
        state_representation = sequence_of_frames.reshape(B, N_frames * D)
        # state_representation = pooled_frames.squeeze(1)
        return state_representation

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, output_DINO: str = 'Attention_Pooling', model_input: str = 'State-Image'):
        super().__init__()

        embed_dim = 384
        self.output_DINO = output_DINO 
        
        if self.output_DINO == 'Attention_Pooling':
            self.backbone = Attention_Pooling(embed_dim, num_heads=4)
        elif self.output_DINO == 'cls':
            self.backbone = nn.Identity()
        
        self.model_input = model_input

        # LazyLinear: dimensione input definita al primo forward
        if self.model_input == 'State-Image':
            self.fc1_state = nn.LazyLinear(128)
            self.fc1_pixels = nn.LazyLinear(256)  # aggiunto perché usi fc1_pixels nel forward
            self.fc2 = nn.LazyLinear(256+128)
            self.norm_layer_fusion = nn.LayerNorm(256+128)
        else:
            self.fc1_pixels = nn.LazyLinear(512)
            self.fc2 = nn.LazyLinear(512)

        self.fc3 = nn.LazyLinear(128)
        self.norm_img = nn.LayerNorm(256)
        self.output_dim = 128

    def init_lazy_weights(self, sample_input):
        """Passa un input dummy per inizializzare i LazyLinear e applicare orthogonal"""
        _ = self.forward(sample_input)  # inizializza tutti i LazyLinear
        list_layer = [self.fc1_pixels, self.fc2, self.fc3]
        if hasattr(self, 'fc1_state'):
            list_layer.append(self.fc1_state)
        for m in list_layer:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 2:
            obs, pixels = inputs
        elif len(inputs) == 3:
            obs, pixels, actions = inputs
            obs = torch.cat([obs, actions], dim = 1)
        
        obs = obs.contiguous()
        pixels = pixels.contiguous()

        x_image = self.backbone(pixels)
        x_image = self.fc1_pixels(x_image)
        x_image = self.norm_img(x_image)
        x_image = F.relu(x_image, inplace=False)
        
        if self.model_input == 'State-Image':
            x_state = self.fc1_state(obs)
            x_state = F.relu(x_state, inplace=False)
            x = torch.cat([x_state, x_image], dim=1)
            x = self.norm_layer_fusion(x)
        else:
            x = x_image

        x = self.fc2(x)
        x = F.relu(x, inplace=False)
        
        x = self.fc3(x)
        x = F.relu(x, inplace=False)

        return x