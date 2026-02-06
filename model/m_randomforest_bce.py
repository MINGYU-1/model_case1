import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEcVAE(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, condition_weights, h1=32, h2=64):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        
        # 가중치를 버퍼로 등록 (모델 저장 시 함께 저장되며, device 이동이 자동화됨)
        # RF에서 계산한 weights_tensor를 여기에 입력합니다.
        self.register_buffer('c_weights', condition_weights)

        ## encoder [x, (c * weights)]
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + c_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1, z_dim)
        self.logvar_head = nn.Linear(h1, z_dim)

        ## decoder_bce [z + (c * weights)]
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim + c_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # 1. Condition에 가중치 적용 (RF에서 얻은 중요도 반영)
        weighted_c = c * self.c_weights
        
        # 2. Encoder: 가중치 적용된 c 사용
        h = self.encoder(torch.cat([x, weighted_c], dim=1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)

        # 3. Decoder: 가중치 적용된 c 사용
        bce_logit = self.decoder_bce(torch.cat([z, weighted_c], dim=1))
        
        return bce_logit, z_mu, z_logvar