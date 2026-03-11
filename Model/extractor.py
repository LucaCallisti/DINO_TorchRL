import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch
from gymnasium import spaces
from torchvision.transforms import v2
import torch.nn.functional as F
from torchrl.envs.transforms import Transform

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase
import numpy as np
from torchrl.data.tensor_specs import UnboundedContinuous
class ExtractorTransform(Transform):
    def __init__(self, device, extractor, dummy_obs_shape = (1, 4, 3, 224, 224), in_keys=None, out_keys=None):
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
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        pixels = tensordict.get(("next", self.in_keys[0]))
        normalized_pixels = self._process(pixels)
        tensordict.set(("next", self.out_keys[0]), normalized_pixels)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        pixels = tensordict_reset.get(self.in_keys[0])
        normalized_pixels = self._process(pixels)
        tensordict_reset.set(self.out_keys[0], normalized_pixels)
        return tensordict_reset

    def _process(self, obs):
        obs_device = obs.device
        with torch.no_grad():
            if next(self.extractor.parameters()).device != obs_device:
                self.extractor.to(obs_device)
                print('spostamento modello gpu')
            for name, param in self.extractor.named_parameters():
                print(name, param.device)
            for name, buffer in self.extractor.named_buffers():
                print(name, buffer.device)
            print("obs.device:", obs.device)
            breakpoint()
            embeddings = self.extractor(obs)
        return embeddings.to(obs_device)

    def transform_observation_spec(self, observation_spec):
        old_spec = observation_spec[self.out_keys[0]]
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
        "vits16_ft": ("dinov3_vits16",  "/home/a.dorizza/GITHUB/DINO_TorchRL/DINO/dino_finetuned_multicrop_200e.pth",  384),
        "vits16":  ("dinov3_vits16",  "DINO/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",  384),
        "vitb16":  ("dinov3_vitb16",  "DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",  768),
    }

    def __init__(self, device: torch.device, model_name: str = "vitb16", output : str = 'cls'):
        super().__init__()
        self.device = device
        self.output = output

        hub_name, weights_path, self.embed_dim = self.MODELS[model_name]

        self.model = torch.hub.load(
            "/home/a.dorizza/GITHUB/DINO_TorchRL/dinov3",
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
            breakpoint()
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
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_frames, N_patch, D = x.shape
        x_flat = x.reshape(B * N_frames, N_patch, D)
        queries = self.query_token.expand(B * N_frames, -1, -1)
        pooled_frames = self.cross_attention(queries, x_flat, x_flat)[0]
        pooled_frames = self.norm(pooled_frames + queries)
        sequence_of_frames = pooled_frames.reshape(B, N_frames, D)
        state_representation = sequence_of_frames.reshape(B, N_frames * D)
        return state_representation

class CNN_backbone(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=32, kernel_size=1),
            nn.ReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=32 * num_frames, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Flatten()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_frames, N_patch, D = x.shape
        H = W = int(N_patch ** 0.5)
        x = x.view(B, N_frames, H, W, D)
        x = x.permute(0, 1, 4, 2, 3)
        x_flat = x.reshape(B * N_frames, D, H, W)

        x_bottle = self.bottleneck(x_flat)

        x_bottle = x_bottle.view(B, N_frames, 32, H, W)
        x_stacked = x_bottle.reshape(B, N_frames * 32, H, W)
        
        cnn_features = self.cnn(x_stacked)
        return cnn_features    

class Model(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_dim : int, n_frame: int, device: torch.device, output_DINO : str = 'cls', model_input : str = 'State-Image'):
        super().__init__(observation_space, features_dim=128)

        self.n_frame = n_frame
        embed_dim = 384
        self.output_DINO = output_DINO
        
        if self.output_DINO == 'Attention_Pooling':
            self.backbone = Attention_Pooling(embed_dim, num_heads=4)
        elif self.output_DINO == 'CNN':
            self.backbone = CNN_backbone(embed_dim, n_frame)
            dummy_input = torch.zeros(1, n_frame, 14*14, embed_dim)
            output = self.backbone(dummy_input)
            input_dim = output.shape[-1]
        elif self.output_DINO == 'cls':
            self.backbone = nn.Identity()
        
        self.model_input = model_input
        self.fc1_pixels = nn.Linear(input_dim, 1024)
        if self.model_input == 'State-Image':
            self.fc1_state = nn.Linear(16, 128)
            self.fc2 = nn.Linear(1024+128, 512)
        else:
            self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

            
        self._init_weights()

    def _init_weights(self):
        list_layer = [self.fc1_pixels, self.fc2, self.fc3]
        if hasattr(self, 'fc1_state'):
            list_layer.append(self.fc1_state)
        for m in list_layer:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x['state']  # B, N_frame, s
        state = state.reshape((state.shape[0], -1)) # B, N_frame*s

        image = x['pixels'] 
        x_image = self.backbone(image)
        x_image = self.fc1_pixels(x_image)
        x_image = F.relu(x_image, inplace = True)

        if self.model_input == 'State-Image':
            x_state = self.fc1_state(state)
            x_state = F.relu(x_state, inplace=True)
            x = torch.cat([x_state, x_image], dim = 1)
        else:
            x = x_image

        x = self.fc2(x)
        x = F.relu(x, inplace = True)
        
        x = self.fc3(x)
        x = F.relu(x, inplace = True)

        return x