import torch
import torch.nn.functional as F

def l_multi3_final_loss(preds, targets, mus, lvs, mode='mse', gamma_list=[0.1, 0.05, 0.02]):
    """
    KL Annealing을 제거하고 계층적 KL Balancing에만 집중한 손실 함수
    """
    batch_size = targets[0].size(0)
    
    # 1. Reconstruction Loss
    recon_loss = 0
    if mode == 'bce':
        for p, t in zip(preds, targets):
            # 존재 여부 학습: BCE with Logits
            recon_loss += F.binary_cross_entropy_with_logits(p, (t > 0).float(), reduction='sum')
    else:
        # 마지막 계층(Pretreatment, index 2)의 MSE 강조 (사용자 지정 1:1:2 가중치)
        recon_loss += F.mse_loss(preds[0], targets[0], reduction='sum') * 0.5
        recon_loss += F.mse_loss(preds[1], targets[1], reduction='sum') * 0.5
        recon_loss += F.mse_loss(preds[2], targets[2], reduction='sum') * 1.0

    # 2. Hierarchical KL Balancing (Annealing 제거)
    kl_total = 0
    for mu, lv, gamma in zip(mus, lvs, gamma_list):
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        # 어닐링 없이 즉시 gamma 가중치 적용
        kl_total += gamma * kl

    return (recon_loss + kl_total) / batch_size
