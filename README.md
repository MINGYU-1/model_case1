# model_case1
![alt text](image.png)

# 분류 및 연속모델 복합 구조 Conditional Variational AutoEncoder(cVAE)를 이용한 건식 개질 공정용 금속 촉매 생성 조성 최적화


본 프로젝트는 건식개질공정 촉매의 물리적/화학적 인과관계를 반영하여, 최적의 촉매 레시피를 제안하는 **계층적 조건부 변분 오토인코더(Hierarchical cVAE)** 모델을 구축하는 것을 목표로 합니다.

# Abstract
 건식 개질 공정은 메탄과 이산화탄소를 합성가스로 전환하는 친환경 기술이지만, 최적의 촉매 조성을 찾기 위한 실험적 비용과 시간적 제약이 크다. 최근 머신러닝 기반의 성능 예측 연구가 활발히 진행되고 있으나, 목표 성능을 만족하는 조성을 도출하기 위해서는 여전히 탐색 범위가 방대하다는 한계가 존재한다. 본 연구에서는 조건을 입력으로 받아 촉매 조성을 직접 생성하는 Conditional Variational AutoEncoder(cVAE) 기반 생성 프레임워크를 제안한다. 특히, 촉매 데이터 세트내 특정 금속의 부재로 발생하는 희소성 문제에 주목하였다. 일반적인 연속 모델은 금속의 존재 유무를 명확히 구분하지 못해 함량 예측에 편향이 발생할 수 있다. 이러한 한계를 보완하기 위해, 금속의 존재 여부를 판단하는 분류 모델과 구체적인 함량을 예측하는 연속 모델을 결합한 복합 구조를 설계하였다. 모델 구조의 타당성을 검증하기 위해 세 가지 아키텍처를 비교하였다. 첫째, 분류 모델과 연속 모델을 완전히 분리하여 학습하는 독립형 병렬구조이다. 둘째, 인코더의 잠재 표현을 공유하는 다중 디코더 구조이다. 셋째, 각 작업에 대해 독립적인 인코더-디코더를 구성하되 동시 학습을 수행하는 구조이다. 또한 분류 모델의 결과를 연속 모델 출력에 반영하는 방식에 따른 성능 차이도 함께 분석하였다. 총 20회 반복 실험 결과, 독립형 병렬 구조에 확률 기반 보정을 적용한 경우 결정계수가 0.8722±0.0118로 가장 우수한 성능을 보였다. 이는 단일 연속 모델의 결정계수 0.8294±0.0285 대비 유의미한 향상이다. 본 연구는 조건부 생성모델을 활용하여 금속 촉매 조성을 체계적으로 생성하는 새로운 설계 전략을 제시한다.

##  최종 목표 (Project Vision)
단순한 수치 예측을 넘어, **[금속 촉매(Metal) → 지지체(Support) → 전처리(Pretreatment)]**로 이어지는 촉매 제조 프로세스의 의사결정 단계를 모방한 생성 모델을 구축합니다. 이를 통해 실제 실험실에서 제조 가능한 수준의 고활성/고안정성 촉매 레시피를 인공지능이 직접 제안(Generative Design)하도록 합니다.

##  모델 구조: Hierarchical cVAE이면서 Nouveau cVAE 
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


##  모델의 최종 목표
이 방식은 **"금속-지지체-전처리" 사이의 강력한 상관관계**를 모델 구조 자체에 이식함으로써, 블랙박스 형태의 AI가 아닌 **물리적 개연성을 가진 촉매 설계안**을 생성할 수 있다는 점에서 차별화됩니다.

## 모델 loss 구하는 방식(KL Balancing)
$$
\mathcal{L}_{total} = \mathbb{E}{q(z|x)} [\log p(x|z)] - \beta \sum_{l=1}^{L} \gamma_l \mathbb{E}{q(z{<l}|x)} [KL(q(z_l|x, z_{<l}) \parallel p(z_l|z_{<l}))]
$$
1. Reconstruction Term ($\mathbb{E}_{q(z|x)} [\log p(x|z)]$): 생성된 촉매 데이터가 실제 실험 데이터와 얼마나 유사한지 측정하는 항입니다 (BCE/MSE).
2. $\beta$ (KL Annealing Factor): 학습 초기에는 0에 가깝게 설정하여 모델이 재구성을 먼저 배우게 하고, 점진적으로 1까지 증가시켜 잠재 공간을 정규화합니다. (KL Annealing)
3. $\gamma_l$ (KL Balancing Coefficient): 본 모델의 핵심 파라미터입니다.
- $\gamma_1 > \gamma_2 > \gamma_3$ 순으로 설정하여 상위 계층(Metal)의 정보 보존을 우선시합니다.
- 이를 통해 하위 계층($z_3$: 전처리)의 KL 항이 너무 강해져서 상위 계층의 정보를 무시하는 'Posterior Collapse' 현상을 방지합니다.
4. $\sum_{l=1}^{L} \dots$ (Hierarchical Structure): 금속($z_1$), 지지체($z_2$), 전처리($z_3$)로 이어지는 계층적 의존 관계를 수학적으로 나타냅니다.
---
## Reference
- https://arxiv.org/pdf/2007.03898
- Kengkanna, A. et al.Reaction-conditioned generative model for catalyst design and optimization withCatDRX. Commun. Chem. 8, 314 (2025)
