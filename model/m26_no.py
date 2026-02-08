import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDecoderCondVAE(nn.Module):
    def __init__(self,x_dim,c_dim,z_dim, h1=32, h2=64): #z1_dim은 다른 encoder에서 넣은값
        
        #z1은 surrogate만들떄 사용
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        
        ## encoder[x,c] 내에서 데이터를 넣는 방법
        self.encoder = nn.Sequential(
            nn.Linear(x_dim+c_dim,h2),
            nn.ReLU(),
            nn.Linear(h2,h1),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(h1,z_dim)
        self.logvar_head = nn.Linear(h1,z_dim)

        ## decoder_bce[z+c]->recon(x_dim)
        self.decoder_bce = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )
        ## decoder_mse[z+c]->recon(x_dim)
        self.decoder_mse = nn.Sequential(
            nn.Linear(z_dim+c_dim,h1),
            nn.ReLU(),
            nn.Linear(h1,h2),
            nn.ReLU(),
            nn.Linear(h2,x_dim)
        )

    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu +std*eps
    
 
    def forward(self,x,c):

        ### 관련해서 z_mu,z_logvar을 활용해서 값을 구하기
        h = self.encoder(torch.cat([x,c],dim = 1))
        z_mu = self.mu_head(h)
        z_logvar = self.logvar_head(h)
        z = self.reparameterize(z_mu,z_logvar)

        bce_logit = self.decoder_bce(torch.concat([z,c],dim = 1))
        prob_mask = torch.sigmoid(bce_logit) ## 확률적 해석
        ## mask_out을 구할때의 방식에 대해 이렇게 구한 값이 0보다 크면 
        binary_out = (prob_mask>0.5).float() ## 결정적 이진 출력
        x_hat = self.decoder_mse(torch.concat([z,c],dim = 1))
        x_hat 
        ## c_에 대한 예측


        return bce_logit ,binary_out, x_hat, z_mu,z_logvar
