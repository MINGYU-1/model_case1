import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    def forward(self, x): return F.relu(x + self.net(x))

class M3_Multi_MSE(nn.Module):
    def __init__(self, x_dims, c_dim, z_dims=[64, 32, 16], h_dim=256):
        super().__init__()
        x1_dim, x2_dim, x3_dim = x_dims
        self.z_dims = z_dims
        
        # Encoders: Residual & BatchNorm 적용으로 학습 안정성 및 정밀도 향상
        self.enc1 = nn.Sequential(nn.Linear(x1_dim + c_dim, h_dim), ResidualBlock(h_dim))
        self.mu1, self.lv1 = nn.Linear(h_dim, z_dims[0]), nn.Linear(h_dim, z_dims[0])
        
        self.enc2 = nn.Sequential(nn.Linear(x2_dim + z_dims[0], h_dim), ResidualBlock(h_dim))
        self.mu2, self.lv2 = nn.Linear(h_dim, z_dims[1]), nn.Linear(h_dim, z_dims[1])
        
        self.enc3 = nn.Sequential(nn.Linear(x3_dim + z_dims[1], h_dim), ResidualBlock(h_dim))
        self.mu3, self.lv3 = nn.Linear(h_dim, z_dims[2]), nn.Linear(h_dim, z_dims[2])

        # Decoders: Sigmoid 제약 추가로 예측값 폭주 방지
        self.dec1 = nn.Sequential(nn.Linear(z_dims[0] + c_dim, h_dim), ResidualBlock(h_dim), nn.Linear(h_dim, x1_dim), nn.Sigmoid())
        self.dec2 = nn.Sequential(nn.Linear(z_dims[1] + z_dims[0], h_dim), ResidualBlock(h_dim), nn.Linear(h_dim, x2_dim), nn.Sigmoid())
        self.dec3 = nn.Sequential(nn.Linear(z_dims[2] + z_dims[1], h_dim), ResidualBlock(h_dim), nn.Linear(h_dim, x3_dim), nn.Sigmoid())

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(self, x1, x2, x3, c):
        h1 = self.enc1(torch.cat([x1, c], 1)); m1, v1 = self.mu1(h1), self.lv1(h1); z1 = self.reparameterize(m1, v1)
        h2 = self.enc2(torch.cat([x2, z1], 1)); m2, v2 = self.mu2(h2), self.lv2(h2); z2 = self.reparameterize(m2, v2)
        h3 = self.enc3(torch.cat([x3, z2], 1)); m3, v3 = self.mu3(h3), self.lv3(h3); z3 = self.reparameterize(m3, v3)
        return [self.dec1(torch.cat([z1, c], 1)), self.dec2(torch.cat([z1, z2], 1)), self.dec3(torch.cat([z2, z3], 1))], [m1, m2, m3], [v1, v2, v3]

    def generate(self, c, device):
        self.eval()
        batch_size = c.size(0)
        with torch.no_grad():
            z1 = torch.randn(batch_size, self.z_dims[0]).to(device)
            z2 = torch.randn(batch_size, self.z_dims[1]).to(device)
            z3 = torch.randn(batch_size, self.z_dims[2]).to(device)
            return [self.dec1(torch.cat([z1, c], 1)), self.dec2(torch.cat([z1, z2], 1)), self.dec3(torch.cat([z2, z3], 1))]
