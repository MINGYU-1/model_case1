import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEcVAE(nn.Module):
    def __init__(self,x_dim,x2_dim, c_dim,z_dim,z2_dim, h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.x2_dim = x2_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.z2_dim = z2_dim
    
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)

         ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)

        ## encoder[x2,z] 내에서 데이터를 만드는 방법
        self.encoder2 = nn.Sequential(
            nn.Linear(x2_dim+z_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu2_head = nn.Linear(h1, z2_dim)
        self.logvar2_head = nn.Linear(h1,z2_dim)

        ## decoder_bce[z+c]->recon(x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim))
        ## decoder2_bce[z2+z]->recon(x_dim)
        self.decoder2_mse = nn.Sequential(
            nn.Linear(z2_dim+z_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x2_dim)
        )

    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps
    
    def reparameterize2(self,mu2,log2_var):
        std2 = torch.exp(0.5*log2_var)
        eps2 = torch.randn_like(std2)
        return mu2 +std2*eps2
        
    def forward(self,x,x2,c):
        h = self.encoder(torch.cat([x, c], dim=1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)
        h2 = self.encoder2(torch.cat([x2, z], dim=1))
        z2_mu = self.mu2_head(h2)
        z2_logvar = self.logvar2_head(h2)
        z2 = self.reparameterize2(z2_mu, z2_logvar)


        x_hat = self.decoder_mse(torch.concat([z,c],dim = 1))
        x2_hat = self.decoder2_mse(torch.concat([z,z2],dim = 1))

        return x2_hat, z_mu, z2_mu,z_logvar, z2_logvar
    

    def inference(self, x2, z, z2=None):
        """
        인퍼런스 단계: 학습된 decoder2_mse를 사용하여 x2_hat을 생성합니다.
        z2가 None이면 0 벡터로 채웁니다.
        주의: 현재 구현에서는 x2 인자는 시그니처 유지용이고, 실제로는 사용하지 않습니다.
        """
        if z2 is None:
            batch_size = z.size(0)
            z2 = torch.zeros(batch_size, self.z2_dim, device=z.device)

        x2_hat = self.decoder2_mse(torch.cat([z, z2], dim=1))
        return x2_hat