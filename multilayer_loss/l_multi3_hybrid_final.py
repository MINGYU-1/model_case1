import torch
import torch.nn.functional as F

def l_multi3_hybrid_final(recon_list, logit_list, target_list, mu_list, lv_list, epoch, total_epochs, gamma_list=[0.1, 0.05, 0.02]):
    """
    BCE + MSE Hybrid 연동 Loss 함수
    """
    batch_size = target_list[0].shape[0]
    total_bce = 0
    total_mse = 0
    
    # 1. Reconstruction Loss (BCE + Masked MSE)
    for hat, logit, target in zip(recon_list, logit_list, target_list):
        # BCE: 존재 여부
        target_bin = (target > 0).float()
        total_bce += F.binary_cross_entropy_with_logits(logit, target_bin, reduction='sum')
        
        # MSE: 존재할 때의 값 (BCE 필터링이 적용된 hat과 비교)
        total_mse += F.mse_loss(hat, target, reduction='sum')

    # 2. KL Annealing & Balancing
    anneal = min(1.0, epoch / (total_epochs * 0.3))
    total_kl = 0
    for mu, lv, gamma in zip(mu_list, lv_list, gamma_list):
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        total_kl += anneal * gamma * kl

    loss = (total_bce + total_mse + total_kl) / batch_size
    
    return {'loss': loss, 'bce': total_bce/batch_size, 'mse': total_mse/batch_size, 'kl': total_kl/batch_size}
