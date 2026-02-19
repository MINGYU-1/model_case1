import torch
import torch.nn.functional as F

def l_multi3_hierarchical(x_hat, x, mu_list, logvar_list, alpha=1.0, gamma_list=[0.1, 0.05, 0.02], step=0, total_steps=1000):
    """
    3단계 계층적 cVAE를 위한 통합 Loss 함수 (Metal -> Support -> Pretreatment)
    
    Args:
        x_hat: 모델의 최종 출력 (Reconstruction)
        x: 실제 타겟 데이터
        mu_list: [mu1, mu2, mu3] 형태의 잠재 변수 평균 리스트
        logvar_list: [logvar1, logvar2, logvar3] 형태의 잠재 변수 로그 분산 리스트
        alpha: Reconstruction Loss 가중치
        gamma_list: [gamma1, gamma2, gamma3] 계층별 KL 가중치 (NVAE balancing 반영)
        step: 현재 학습 step (Annealing용)
        total_steps: KL Annealing을 완료할 총 step 수
    """
    batch_size = x.shape[0]

    # 1. Masked MSE Reconstruction Loss
    # 성분이 존재하는 곳(x > 0)의 수치 정밀도에 집중
    mask = (x > 0).float()
    recon_loss = torch.sum(mask * (x_hat - x)**2)

    # 2. KL Annealing Coefficient 계산 (Linear Annealing)
    # 초기에는 0에서 시작하여 total_steps에 걸쳐 1까지 증가
    anneal_weight = min(1.0, step / total_steps)

    # 3. 계층별 KL Divergence 계산 및 Balancing
    # gamma_list는 보통 [상위(큰값), 중간, 하위(작은값)] 순서로 설정 (NVAE trend)
    kl_losses = []
    total_weighted_kl = 0
    
    for i in range(len(mu_list)):
        mu = mu_list[i]
        logvar = logvar_list[i]
        gamma = gamma_list[i]
        
        # 기본 KL 수식
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_losses.append(kl / batch_size)
        
        # Annealing과 계층별 가중치 적용
        total_weighted_kl += anneal_weight * gamma * kl

    # 4. 최종 통합 손실
    total_loss = (alpha * recon_loss + total_weighted_kl) / batch_size

    return {
        'loss': total_loss,
        'recon_loss': recon_loss / batch_size,
        'kl_metal': kl_losses[0] if len(kl_losses) > 0 else 0,
        'kl_support': kl_losses[1] if len(kl_losses) > 1 else 0,
        'kl_pretreat': kl_losses[2] if len(kl_losses) > 2 else 0,
        'anneal_w': anneal_weight
    }

def l_multi3_bce_hierarchical(binary_logit, x, mu_list, logvar_list, beta=1.0, gamma_list=[0.1, 0.05, 0.02], step=0, total_steps=1000):
    """
    3단계 계층적 cVAE를 위한 존재 여부(BCE) 기반 통합 Loss 함수
    """
    batch_size = x.shape[0]
    
    # 1. Classification Loss (BCE)
    x_binary = (x > 0).float()
    bce_loss = F.binary_cross_entropy_with_logits(binary_logit, x_binary, reduction='sum')

    # 2. KL Annealing Coefficient
    anneal_weight = min(1.0, step / total_steps)

    # 3. 계층별 KL Balancing
    total_weighted_kl = 0
    kl_values = []
    
    for i in range(len(mu_list)):
        kl = -0.5 * torch.sum(1 + logvar_list[i] - mu_list[i].pow(2) - logvar_list[i].exp())
        kl_values.append(kl / batch_size)
        total_weighted_kl += anneal_weight * gamma_list[i] * kl

    total_loss = (beta * bce_loss + total_weighted_kl) / batch_size

    return {
        'loss': total_loss,
        'bce_loss': bce_loss / batch_size,
        'kl_metal': kl_values[0],
        'kl_support': kl_values[1],
        'kl_pretreat': kl_values[2],
        'anneal_w': anneal_weight
    }
