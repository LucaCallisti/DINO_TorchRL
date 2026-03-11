import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces
from torchvision.transforms import v2
import torch.nn.functional as F

            
class VecExtractorWrapper(VecEnvWrapper):
    def __init__(self, venv, extractor):
        super().__init__(venv)
        self.extractor = extractor
        dummy_obs = torch.zeros((1, 4, 3, 224, 224)).to(extractor.device)   # 4 stacked frame
        with torch.no_grad():
            dummy_output = self.extractor(dummy_obs)
        self.observation_space['pixels'] = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=dummy_output.shape[1:], 
            dtype=np.float32
        )
        self.observation_space['state']=spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4,), dtype=np.float32)   # il primo 4 è per il numero di frame
    def reset(self):
        obs = self.venv.reset()
        return self._process_obs(obs)
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            if "terminal_observation" in info:
                last_obs = {k : np.expand_dims(v, axis = 0) for k, v in info['terminal_observation'].items()}
                processed_obs = self._process_obs(last_obs)
                info['terminal_observation'] = {k: v.squeeze(0) for k, v in processed_obs.items()}                

        return self._process_obs(obs), rewards, dones, infos
    def _process_obs(self, obs):
        pixels_tensor = torch.from_numpy(obs['pixels']).to(self.extractor.device, non_blocking=True)    # B, N_frame, H, W, C
        pixels_tensor = self._resize_images(pixels_tensor)
        with torch.no_grad():
            with torch.autocast(device_type=self.extractor.device.type, dtype=torch.bfloat16):
                embeddings = self.extractor(pixels_tensor)
        obs = obs.copy() # Avoid modifying original dict if shared
        obs['pixels'] = embeddings.float().cpu().numpy().astype(np.float32)
        obs['state'] = obs['state'][:, :, :4].astype(np.float32)
        return obs

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        '''
        images: (B, N_frame, H, W, C)
        return: (B*N_frame, C, 224, 224)
        '''
        images = images.float() / 255.0
        images = images.permute(0, 1, 4, 2, 3).contiguous()   # B, N_frame, C, H, W
        old_shape = images.shape
        images = images.reshape((-1, *images.shape[2:]))   # B*N_frame, C, H, W
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        return images.reshape(old_shape[0], old_shape[1], *images.shape[1:])
    
    
class DinoExtractor(nn.Module):
    MODELS = {
        "vits16_ft": ("dinov3_vits16",  "DINO/dino_finetuned_multicrop_200e.pth",  384),
        "vits16":  ("dinov3_vits16",  "DINO/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",  384),
        "vitb16":  ("dinov3_vitb16",  "DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",  768),
    }

    def __init__(self, device: torch.device, model_name: str = "vitb16", output : str = 'cls'):
        super().__init__()
        self.device = device
        self.output = output

        hub_name, weights_path, self.embed_dim = self.MODELS[model_name]

        self.model = torch.hub.load(
            "DINO/dinov3",
            hub_name,
            source="local",
            weights=weights_path
        ).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = torch.compile(self.model)

        def make_transform():
            normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            return v2.Compose([normalize])
        self.transform = make_transform()
    
    def forward(self, img):
        '''
        img: (B, N_frames, C, H, W)
        '''
        with torch.no_grad():
            init_shape = img.shape # (B, N_frames, C, H, W)
            image = img.reshape(-1, init_shape[-3], init_shape[-2], init_shape[-1])
            img_tensor = self.transform(image)
            features = self.model.forward_features(img_tensor, masks=None)
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