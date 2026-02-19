import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCVAE3(nn.Module):
    """
    3단계 계층적 cVAE (Metal -> Support -> Pretreatment)
    각 단계는 이전 단계의 잠재 변수(z)를 조건으로 사용합니다.
    """
    def __init__(self, x_dims, c_dim, z_dims, h_dims=[64, 32]):
        """
        Args:
            x_dims: [x1_dim, x2_dim, x3_dim] 각 계층 데이터 차원
            c_dim: 외부 조건(condition) 차원 (예: 온도, 압력 등)
            z_dims: [z1_dim, z2_dim, z3_dim] 각 계층 잠재 변수 차원
            h_dims: 은닉층 차원 [h_large, h_small]
        """
        super().__init__()
        x1_dim, x2_dim, x3_dim = x_dims
        z1_dim, z2_dim, z3_dim = z_dims
        h2, h1 = h_dims
        
        self.z_dims = z_dims

        # --- Encoder 1: [Metal(x1) + Condition(c)] -> z1 ---
        self.enc1 = nn.Sequential(
            nn.Linear(x1_dim + c_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU()
        )
        self.mu1 = nn.Linear(h1, z1_dim)
        self.var1 = nn.Linear(h1, z1_dim)

        # --- Encoder 2: [Support(x2) + z1] -> z2 ---
        self.enc2 = nn.Sequential(
            nn.Linear(x2_dim + z1_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU()
        )
        self.mu2 = nn.Linear(h1, z2_dim)
        self.var2 = nn.Linear(h1, z2_dim)

        # --- Encoder 3: [Pretreatment(x3) + z2] -> z3 ---
        self.enc3 = nn.Sequential(
            nn.Linear(x3_dim + z2_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU()
        )
        self.mu3 = nn.Linear(h1, z3_dim)
        self.var3 = nn.Linear(h1, z3_dim)

        # --- Decoder 1 (BCE): [z1 + c] -> x1_prob (Metal Presence) ---
        self.dec1_bce = nn.Sequential(
            nn.Linear(z1_dim + c_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x1_dim)
        )

        # --- Decoder 2 (MSE): [z1 + z2] -> x2_hat (Support) ---
        self.dec2_mse = nn.Sequential(
            nn.Linear(z1_dim + z2_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x2_dim)
        )

        # --- Decoder 3 (MSE): [z2 + z3] -> x3_hat (Pretreatment) ---
        self.dec3_mse = nn.Sequential(
            nn.Linear(z2_dim + z3_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, x3_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x1, x2, x3, c):
        # Encoding Step 1
        h1 = self.enc1(torch.cat([x1, c], dim=1))
        mu1, lv1 = self.mu1(h1), self.var1(h1)
        z1 = self.reparameterize(mu1, lv1)

        # Encoding Step 2 (z1에 의존)
        h2 = self.enc2(torch.cat([x2, z1], dim=1))
        mu2, lv2 = self.mu2(h2), self.var2(h2)
        z2 = self.reparameterize(mu2, lv2)

        # Encoding Step 3 (z2에 의존)
        h3 = self.enc3(torch.cat([x3, z2], dim=1))
        mu3, lv3 = self.mu3(h3), self.var3(h3)
        z3 = self.reparameterize(mu3, lv3)

        # Decoding
        x1_logit = self.dec1_bce(torch.cat([z1, c], dim=1))
        x2_hat = self.dec2_mse(torch.cat([z1, z2], dim=1))
        x3_hat = self.dec3_mse(torch.cat([z2, z3], dim=1))

        return {
            'x1_logit': x1_logit, 'x2_hat': x2_hat, 'x3_hat': x3_hat,
            'mu_list': [mu1, mu2, mu3],
            'lv_list': [lv1, lv2, lv3]
        }

    def sample(self, c, num_samples=1, device='cpu'):
        """
        Generation: 조건 c만으로 [금속 -> 지지체 -> 전처리]를 순차적으로 생성합니다.
        """
        self.eval()
        with torch.no_grad():
            # 1. Metal Generation (z1)
            z1 = torch.randn(num_samples, self.z_dims[0]).to(device)
            x1_logit = self.dec1_bce(torch.cat([z1, c], dim=1))
            x1_prob = torch.sigmoid(x1_logit)

            # 2. Support Generation (z2 conditioned on z1)
            z2 = torch.randn(num_samples, self.z_dims[1]).to(device)
            x2_hat = self.dec2_mse(torch.cat([z1, z2], dim=1))

            # 3. Pretreatment Generation (z3 conditioned on z2)
            z3 = torch.randn(num_samples, self.z_dims[2]).to(device)
            x3_hat = self.dec3_mse(torch.cat([z2, z3], dim=1))

        return x1_prob, x2_hat, x3_hat
