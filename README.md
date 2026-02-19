# model_case1
![alt text](image.png)


# Hierarchical cVAE for Dry Reforming of Methane (DRM) Catalyst Design

본 프로젝트는 건식개질공정(DRM) 촉매의 물리적/화학적 인과관계를 반영하여, 최적의 촉매 레시피를 제안하는 **계층적 조건부 변분 오토인코더(Hierarchical cVAE)** 모델을 구축하는 것을 목표로 합니다.

##  최종 목표 (Project Vision)
단순한 수치 예측을 넘어, **[금속 촉매(Metal) → 지지체(Support) → 전처리(Pretreatment)]**로 이어지는 촉매 제조 프로세스의 의사결정 단계를 모방한 생성 모델을 구축합니다. 이를 통해 실제 실험실에서 제조 가능한 수준의 고활성/고안정성 촉매 레시피를 인공지능이 직접 제안(Generative Design)하도록 합니다.

##  모델 구조: Hierarchical cVAE
본 모델은 일반적인 단일 잠재 공간(Latent Space) 모델과 달리, 도메인 지식을 기반으로 한 **3단계 계층 구조**를 채택하고 있습니다.

### 1. 계층적 의존성 (Hierarchical Dependency)
*   **Step 1 (Metal Group):** 금속 성분의 종류와 함량을 잠재 변수 $z$로 인코딩합니다.
*   **Step 2 (Support Group):** 앞서 결정된 금속($z$)을 조건으로, 최적의 지지체 조합을 잠재 변수 $z_2$로 학습합니다.
*   **Step 3 (Pretreatment Group):** 금속과 지지체의 조합($z, z_2$)을 모두 고려하여, 최적의 전처리 조건($z_3$)을 결정합니다.

### 2. 하이브리드 디코딩 (BCE + MSE Decoder)
*   **BCE Decoder:** 특정 성분의 포함 여부(Classification)를 결정하여 생성물의 구조적 타당성을 확보합니다.
*   **MSE Decoder:** 포함된 성분의 구체적인 함량 및 반응 수치(Regression)를 정밀하게 예측합니다.

##  데이터 전처리 및 검증 지표
*   **Dataset:** `211210-DRM-total.csv` (건식개질 촉매 실험 데이터)
*   **Scaling:** 지지체 비중을 기준으로 한 **Min-Max Scaling**을 적용하여 실제 제조 레시피의 상대적 비율을 유지합니다.
*   **Key Metrics:**
    *   **$R^2$ Score:** 수치적 재구성 정확도 측정
    *   **Catalyst Feasibility:** 생성된 레시피가 지지체 대비 금속 함량의 물리적 상한선 및 공학적 타당성을 만족하는지 검증 (Domain-specific validation)

##  주요 파일 
*   `MODEL_CASE1/multilayer_rand_st/18_multi.ipynb`: 계층별 개별 Loss(Multilayer Loss)를 계산하고 최적화하는 메인 학습 파이프라인.
*   `MSEcVAE 클래스, BCEcVAE 클래스`: 계층적 잠재 공간과 다중 디코더가 통합된 커스텀 PyTorch 모델 객체.

---

##  모델의 특장점
이 방식은 **"금속-지지체-전처리" 사이의 강력한 상관관계**를 모델 구조 자체에 이식함으로써, 블랙박스 형태의 AI가 아닌 **물리적 개연성을 가진 촉매 설계안**을 생성할 수 있다는 점에서 차별화됩니다.

---