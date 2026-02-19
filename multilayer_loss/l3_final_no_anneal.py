import torch
import torch.nn.functional as F

def l3_final_loss_no_anneal(preds, targets, mus, lvs, mode='mse', gamma_list=[0.01, 0.005, 0.002]):
    """
    R2 점수를 높이기 위해 KL 가중치를 낮게 설정하고 어닐링을 제거함
    """
    batch_size = targets[0].size(0)
    recon_loss = 0
    
    if mode == 'bce':
        for p, t in zip(preds, targets):
            recon_loss += F.binary_cross_entropy_with_logits(p, (t > 0).float(), reduction='sum')
    else:
        # 마지막 계층(Pretreat)과 금속 계층(Metal)의 비중을 높임
        recon_loss += F.mse_loss(preds[0], targets[0], reduction='sum') * 1.0 # Metal
        recon_loss += F.mse_loss(preds[1], targets[1], reduction='sum') * 1.0 # Support
        recon_loss += F.mse_loss(preds[2], targets[2], reduction='sum') * 1.0 # Pretreat

    # KL Balancing (Immediately Applied)
    kl_total = 0
    for mu, lv, gamma in zip(mus, lvs, gamma_list):
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        kl_total += gamma * kl

    return (recon_loss + kl_total) / batch_size
