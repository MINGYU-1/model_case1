import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalHybridCVAE3(nn.Module):
    """
    3단계 계층적 Hybrid cVAE (Metal -> Support -> Pretreatment)
    각 Decoder 단계마다 BCE(존재 확률)와 MSE(수치)를 연동하여 출력합니다.
    """
    def __init__(self, x_dims, c_dim, z_dims, h_dims=[128, 64]):
        super().__init__()
        x1_dim, x2_dim, x3_dim = x_dims
        z1_dim, z2_dim, z3_dim = z_dims
        h2, h1 = h_dims
        self.z_dims = z_dims

        # --- Encoder Chain ---
        self.enc1 = nn.Sequential(nn.Linear(x1_dim + c_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu1, self.var1 = nn.Linear(h1, z1_dim), nn.Linear(h1, z1_dim)

        self.enc2 = nn.Sequential(nn.Linear(x2_dim + z1_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu2, self.var2 = nn.Linear(h1, z2_dim), nn.Linear(h1, z2_dim)

        self.enc3 = nn.Sequential(nn.Linear(x3_dim + z2_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu3, self.var3 = nn.Linear(h1, z3_dim), nn.Linear(h1, z3_dim)

        # --- Hybrid Decoders (BCE + MSE) ---
        # Metal (Phase 1)
        self.dec1_bce = nn.Sequential(nn.Linear(z1_dim + c_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x1_dim))
        self.dec1_mse = nn.Sequential(nn.Linear(z1_dim + c_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x1_dim))

        # Support (Phase 2)
        self.dec2_bce = nn.Sequential(nn.Linear(z1_dim + z2_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x2_dim))
        self.dec2_mse = nn.Sequential(nn.Linear(z1_dim + z2_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x2_dim))

        # Pretreatment (Phase 3)
        self.dec3_bce = nn.Sequential(nn.Linear(z2_dim + z3_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x3_dim))
        self.dec3_mse = nn.Sequential(nn.Linear(z2_dim + z3_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x3_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2, x3, c):
        h1 = self.enc1(torch.cat([x1, c], dim=1)); mu1, lv1 = self.mu1(h1), self.var1(h1); z1 = self.reparameterize(mu1, lv1)
        h2 = self.enc2(torch.cat([x2, z1], dim=1)); mu2, lv2 = self.mu2(h2), self.var2(h2); z2 = self.reparameterize(mu2, lv2)
        h3 = self.enc3(torch.cat([x3, z2], dim=1)); mu3, lv3 = self.mu3(h3), self.var3(h3); z3 = self.reparameterize(mu3, lv3)

        # Hybrid Output Logic
        def get_hybrid(bce_layer, mse_layer, latent):
            logit = bce_layer(latent)
            raw = mse_layer(latent)
            prob = torch.sigmoid(logit)
            # BCE 확률을 MSE 값에 곱하여 '0'인 부분을 확실히 필터링
            return prob * raw, logit

        x1_hat, x1_logit = get_hybrid(self.dec1_bce, self.dec1_mse, torch.cat([z1, c], dim=1))
        x2_hat, x2_logit = get_hybrid(self.dec2_bce, self.dec2_mse, torch.cat([z1, z2], dim=1))
        x3_hat, x3_logit = get_hybrid(self.dec3_bce, self.dec3_mse, torch.cat([z2, z3], dim=1))

        return {
            'recons': [x1_hat, x2_hat, x3_hat],
            'logits': [x1_logit, x2_logit, x3_logit],
            'mus': [mu1, mu2, mu3], 'lvs': [lv1, lv2, lv3]
        }

    def sample(self, c, device='cpu'):
        self.eval()
        with torch.no_grad():
            z1 = torch.randn(c.size(0), self.z_dims[0]).to(device)
            p1 = torch.sigmoid(self.dec1_bce(torch.cat([z1, c], dim=1))) * self.dec1_mse(torch.cat([z1, c], dim=1))
            
            z2 = torch.randn(c.size(0), self.z_dims[1]).to(device)
            p2 = torch.sigmoid(self.dec2_bce(torch.cat([z1, z2], dim=1))) * self.dec2_mse(torch.cat([z1, z2], dim=1))
            
            z3 = torch.randn(c.size(0), self.z_dims[2]).to(device)
            p3 = torch.sigmoid(self.dec3_bce(torch.cat([z2, z3], dim=1))) * self.dec3_mse(torch.cat([z2, z3], dim=1))
        return p1, p2, p3
