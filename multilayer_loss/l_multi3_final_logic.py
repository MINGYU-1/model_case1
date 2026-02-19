import torch
import torch.nn.functional as F

def l_multi3_final_loss(preds, targets, mus, lvs, epoch, total_epochs, mode='mse', gamma_list=[0.1, 0.05, 0.02]):
    batch_size = targets[0].size(0)
    
    # 1. Reconstruction Loss
    recon_loss = 0
    if mode == 'bce':
        for p, t in zip(preds, targets):
            recon_loss += F.binary_cross_entropy_with_logits(p, (t > 0).float(), reduction='sum')
    else:
        # 마지막 계층(Pretreatment, index 2)의 MSE 강조
        # 다른 계층도 학습하되, 마지막 계층에 가중치를 더 줌 (예: 1:1:2)
        recon_loss += F.mse_loss(preds[0], targets[0], reduction='sum') * 0.5
        recon_loss += F.mse_loss(preds[1], targets[1], reduction='sum') * 0.5
        recon_loss += F.mse_loss(preds[2], targets[2], reduction='sum') * 1.0 # Last layer emphasis

    # 2. KL Balancing (NVAE)
    anneal = min(1.0, epoch / (total_epochs * 0.3))
    kl_total = 0
    for mu, lv, gamma in zip(mus, lvs, gamma_list):
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        kl_total += anneal * gamma * kl

    return (recon_loss + kl_total) / batch_size
