import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEcVAE(nn.Module):
    def __init__(self,x_dim,x2_dim, c_dim,z_dim,z2_dim,  h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
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

        ## encoder[x2,z] 내에서 데이터를 만드는 방법
        self.encoder2 = nn.Sequential(
            nn.Linear(x2_dim+z_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu2_head = nn.Linear(h1, z2_dim)
        self.logvar_head = nn.Linear(h1,z2_dim)

        ## decoder_bce[z+c]->recon(x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim))
        ## decoder2_bce[z2+z]->recon(x_dim)
        self.decoder2_bce = nn.Sequential(
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
        h2 = self.encoder2(torch.cat([x2,z],dim = 1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu, z_logvar)
        z2_mu = self.mu2_head(h)
        z2_logvar = self.logvar2_head(h)
        z2 = self.reparameterize2(z2_mu,z2_logvar)
        bce_logit = self.decoder_bce(torch.concat([z, c], dim=1))
        bce2_logit = self.decoder2_bce(torch.concat([z2,z],dim = 1))
        return bce2_logit, z_mu, z_logvar
    
