import torch
import torch.nn as nn

class M3_Multi_BCE(nn.Module):
    def __init__(self, x_dims, c_dim, z_dims=[16, 8, 4], h_dims=[128, 64]):
        super().__init__()
        x1_dim, x2_dim, x3_dim = x_dims
        z1_dim, z2_dim, z3_dim = z_dims
        h2, h1 = h_dims
        self.z_dims = z_dims
        
        # Encoders
        self.enc1 = nn.Sequential(nn.Linear(x1_dim + c_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu1, self.lv1 = nn.Linear(h1, z1_dim), nn.Linear(h1, z1_dim)
        self.enc2 = nn.Sequential(nn.Linear(x2_dim + z1_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu2, self.lv2 = nn.Linear(h1, z2_dim), nn.Linear(h1, z2_dim)
        self.enc3 = nn.Sequential(nn.Linear(x3_dim + z2_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU())
        self.mu3, self.lv3 = nn.Linear(h1, z3_dim), nn.Linear(h1, z3_dim)

        # Decoders (존재 여부 로짓 출력)
        self.dec1 = nn.Sequential(nn.Linear(z1_dim + c_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x1_dim))
        self.dec2 = nn.Sequential(nn.Linear(z1_dim + z2_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x2_dim))
        self.dec3 = nn.Sequential(nn.Linear(z2_dim + z3_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, x3_dim))

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
            o1 = self.dec1(torch.cat([z1, c], 1))
            o2 = self.dec2(torch.cat([z1, z2], 1))
            o3 = self.dec3(torch.cat([z2, z3], 1))
        return [o1, o2, o3]
