import torch
import torch.nn.functional as F

def l_multi3_final_loss(preds, targets, mus, lvs, mode='mse', gamma_list=[0.01, 0.005, 0.002]):
    """
    NVAE(2020) 기반 KL Balancing 및 1:1:1 Reconstruction 가중치 적용
    """
    batch_size = targets[0].size(0)
    recon_loss = 0
    
    if mode == 'bce':
        for p, t in zip(preds, targets):
            # 모든 계층 동일하게 1:1:1 가중치 적용
            recon_loss += F.binary_cross_entropy_with_logits(p, (t > 0).float(), reduction='sum')
    else:
        # MSE Reconstruction 가중치 1:1:1 적용
        recon_loss += F.mse_loss(preds[0], targets[0], reduction='sum') * 1.0
        recon_loss += F.mse_loss(preds[1], targets[1], reduction='sum') * 1.0
        recon_loss += F.mse_loss(preds[2], targets[2], reduction='sum') * 1.0

    # KL Balancing 가중치 적용 [0.01, 0.005, 0.002]
    kl_total = 0
    for mu, lv, gamma in zip(mus, lvs, gamma_list):
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        kl_total += gamma * kl

    return (recon_loss + kl_total) / batch_size
