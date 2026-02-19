import os
import numpy as np
import pandas as pd
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
# 사용자 정의 모듈 경로 설정 유지
from vae_earlystopping import EarlyStopping
from model.case1_bce import BCEcVAE
from model.case1_mse import MSEcVAE
from loss.l2_bce import l2_bce
from loss.l2_mse import l2_mse

results = {
    "random_state": [],
    "R2_MSE": [],           # 순수 MSE 결과
    "R2_HARD_LABEL": [],    # bce_binary 곱한 후 역변환
    "R2_SOFT_LABEL": []     # bce_prob 곱한 후 역변환
}

# 20회 반복 실험
np.random.seed(42)
random_seeds = np.random.randint(1, 100, size=20)

for i in random_seeds:
    print(f"\n--- Processing Random State: {i} ---")
    
    # 1. 데이터 로드 및 전처리
    x_path = os.path.join(base_path, 'data', 'metal.npy')
    c_path = os.path.join(base_path, 'data', 'pre_re_change_temp_logconst.npy')
    if not os.path.exists(x_path):
        print(f"❌ 에러: {x_path} 파일을 찾을 수 없습니다! 폴더 구조를 확인하세요.")
    else:
        x_data = np.load(x_path)
        c_data = np.load(c_path)
        print("✅ 데이터 로드 성공!")
    x_train, x_test, c_train, c_test = train_test_split(x_data, c_data, random_state=i, test_size=0.4)
    x_val, x_test, c_val, c_test = train_test_split(x_test, c_test, random_state=i, test_size=0.5)
    
    x_scaler, c_scaler = MinMaxScaler(), MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    c_train = c_scaler.fit_transform(c_train)
    x_val, x_test = [x_scaler.transform(x) for x in [x_val, x_test]]
    c_val, c_test = [c_scaler.transform(c) for c in [c_val, c_test]]

    x_train, x_val, x_test = [torch.tensor(x, dtype=torch.float32) for x in [x_train, x_val, x_test]]
    c_train, c_val, c_test = [torch.tensor(c, dtype=torch.float32) for c in [c_train, c_val, c_test]]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(TensorDataset(x_train, c_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, c_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, c_test), batch_size=64, shuffle=False)

    x_dim, c_dim = x_train.shape[1], c_train.shape[1]

    # 2. BCE 모델 학습
    bce_model = BCEcVAE(x_dim, c_dim, z_dim=8).to(device)
    early_stop_bce = EarlyStopping(patience=40, min_delta=1e-9)
    opt_bce = optim.Adam(bce_model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(1, 801):
        bce_model.train()
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt_bce.zero_grad()
            logits, mu, logvar = bce_model(x, c)
            loss_dict = l2_bce(logits, x, mu, logvar)
            loss_dict['loss'].backward()
            opt_bce.step()
        
        # Validation (단순화된 형태)
        bce_model.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vc in val_loader:
                vx, vc = vx.to(device), vc.to(device)
                vl, vm, vlv = bce_model(vx, vc)
                v_loss += l2_bce(vl, vx, vm, vlv)['loss'].item()
        if early_stop_bce(v_loss/len(val_loader), bce_model): break

    # 3. MSE 모델 학습
    mse_model = MSEcVAE(x_dim, c_dim, z_dim=8).to(device)
    early_stop_mse = EarlyStopping(patience=40, min_delta=1e-9)
    opt_mse = optim.Adam(mse_model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(1, 801):
        mse_model.train()
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            opt_mse.zero_grad()
            x_hat, mu, logvar = mse_model(x, c)
            loss_dict = l2_mse(x_hat, x, mu, logvar)
            loss_dict['loss'].backward()
            opt_mse.step()
            
        mse_model.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vc in val_loader:
                vx, vc = vx.to(device), vc.to(device)
                vh, vm, vlv = mse_model(vx, vc)
                v_loss += l2_mse(vh, vx, vm, vlv)['loss'].item()
        if early_stop_mse(v_loss/len(val_loader), mse_model): break

    # 4. 추론 및 필터링 후 역변환 (순서: Mask -> Inverse)
    bce_model.eval()
    mse_model.eval()
    
    all_bce_logits = []
    all_mse_scaled = []
    all_x_true_scaled = []

    with torch.no_grad():
        for xt, ct in test_loader:
            xt, ct = xt.to(device), ct.to(device)
            b_logits, _, _ = bce_model(xt, ct)
            m_hat, _, _ = mse_model(xt, ct)
            all_bce_logits.append(b_logits.cpu().numpy())
            all_mse_scaled.append(m_hat.cpu().numpy())
            all_x_true_scaled.append(xt.cpu().numpy())

    bce_logits_np = np.vstack(all_bce_logits)
    mse_scaled_np = np.vstack(all_mse_scaled)
    x_true_scaled_np = np.vstack(all_x_true_scaled)

    # [순서 핵심] 1. 스케일링 된 상태(0~1)에서 BCE 마스크 적용
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    bce_prob = sigmoid(bce_logits_np)
    bce_binary = (bce_prob >= 0.5).astype(np.float32)

    mse_hard_scaled = mse_scaled_np * bce_binary
    mse_soft_scaled = mse_scaled_np * bce_prob

    # [순서 핵심] 2. 필터링된 결과들을 각각 역변환
    x_true_inv = x_scaler.inverse_transform(x_true_scaled_np)
    x_hat_mse_inv = x_scaler.inverse_transform(mse_scaled_np)
    x_hat_hard_inv = x_scaler.inverse_transform(mse_hard_scaled)
    x_hat_soft_inv = x_scaler.inverse_transform(mse_soft_scaled)

    # 5. 성능 평가
    r2_mse = r2_score(x_true_inv.flatten(), x_hat_mse_inv.flatten())
    r2_hard = r2_score(x_true_inv.flatten(), x_hat_hard_inv.flatten())
    r2_soft = r2_score(x_true_inv.flatten(), x_hat_soft_inv.flatten())

    results["random_state"].append(int(i))
    results["R2_MSE"].append(float(r2_mse))
    results["R2_HARD_LABEL"].append(float(r2_hard))
    results["R2_SOFT_LABEL"].append(float(r2_soft))

# JSON 저장
save_path = "./results_case1_logscale3cases.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n실험 완료! 결과 저장됨: {save_path}")