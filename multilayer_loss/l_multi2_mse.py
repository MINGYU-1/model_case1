import torch
import torch.nn.functional as F
def l2_mse(x_hat, x, mu,mu2, logvar, logvar2,
                       alpha=1.0,gamma1=0.1,gamma2=0.05):


    mse_loss = F.mse_loss(x_hat, x, reduction='sum')

    kl_loss = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    kl2_loss = -0.5*torch.sum(1+logvar2-mu2.pow(2)-logvar2.exp())
    batch_size = x.shape[0]
    total_loss = (alpha * mse_loss + gamma1 * kl_loss+ gamma2 * kl2_loss)/batch_size
    
    return {
        'loss': total_loss,
        'mse_loss': mse_loss/batch_size ,
        'kl_loss': kl_loss/batch_size,
        'kl2_loss':kl2_loss/batch_size
    }