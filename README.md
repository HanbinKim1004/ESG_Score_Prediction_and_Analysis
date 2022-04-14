# 22-1_DSL_Modeling_Clustering_Algorithm

file definition

code:
1. saramin_crawling = 사람인 크롤링 데이터 
2. clusteringrepeat = oversampling clustering / cluster 속 clusterting
3. 0000 = 
4. 0000 = 

data:
1. 복지 (1).csv = 사람인 크롤링 데이터

![image](https://user-images.githubusercontent.com/77422840/163293893-9b4effce-1e45-4ac6-9548-3ef11cf730db.png)


![image](https://user-images.githubusercontent.com/77422840/163296786-8ed17110-6b84-453f-8c7b-6289a69d0130.png)

문제점: 
1. 한국 ESG 등급 산정의 실태는 굉장히 열악함
- 기관별로 ESG 등급 산정 Feature 및 가중치 설정이 상이함

- 기관별로 ESG 등급 산정 시 재무 정보와 비재무정보가 무차별적으로 섞여있어서, 등급 자체의 신뢰도가 떨어지는 편

- 상장 기업 중에서도 ESG 등급이 산정되지 않은 기업이 존재

![image](https://user-images.githubusercontent.com/77422840/163296923-7e9ff8de-0174-4bd8-9b3b-4070fa394fc5.png)

데이터 전처리1.  Label Encoding (Ordinal Encoding)
- 각 문자열 라벨을 단순 정수화 (A -> 0, B -> 1, C -> 2…)
- Decision Tree 모델에 적합하며, 특히 GBDT 모델에서는 기본적인 방법
- 클러스터링의 경우에는 범주형 자료를 숫자형으로 변환하지 않고 진행


![image](https://user-images.githubusercontent.com/77422840/163297128-c9757fcb-dd5b-4f47-9e65-7e685c64d614.png)

데이터 전처리1.  2.  SMOTE(Synthetic Minority Over-Sampling Technique) – 데이터 비대칭 문제 해결

- 비대칭 문제: 데이터의 비율 차이가 큰 경우 단순히 우세한 클래스의 모형을 선택하는 모형의 정확도가 높아짐
- SMOTE : 낮은 비율로 존재하는 클래스의 데이터를 K-NN 알고리즘(최근접)을 활용하여 새롭게 생성하는 방법. 
- 단순 무작위 추출은 overfitting 문제가 발생할 수도 있으나, SMOTE는 알고리즘에 기반해서 데이터를 생성하므로 과적합 발생 가능성이 상대적으로 작음

![image](https://user-images.githubusercontent.com/77422840/163297245-0c83a988-76ac-488b-ba91-cb8385e63758.png)
모델1 - Unsupervised Learning – K-Prototype Model (군집화 및 해석 목적)

- K-Prototype Clustering은 K-means(수치형 자료, 평균값 사용) + K-modes(범주형 자료, 최빈값 사용)	
- 두 모델을 동시에 사용함으로써, 범주형 자료의 변환없이 클러스터링이 가능한 알고리즘

![image](https://user-images.githubusercontent.com/77422840/163297419-b0e893d7-da9a-4836-b26f-20eb86fa269f.png)

모델2 - Supervised Learning - Decision Tree, Random Forest, ADA Boost, Light GBM, Gradient Boosting, CAT Boosting, MLP (총 7개 Model)
 	- 비지도학습 기반의 클러스터링 모델은 기업이 속한 군집의 속성을 바탕으로 ESG 등급을 확률로서 예측함

	- 지도학습에 기반한 Classification의 성능의 테스트, Y factor에 가장 큰 영향을 주는 X feature의 탐색을 위함

	- F1 Score을 활용해 성능 평가

	- ESG 등급을 제외한 모델의 지도학습으로 얻은 Label과 실제 Label을 비교하여 예측 성능 평가

	- 변수 중요도 확인으로 가장 많은 영향을 끼치는 Feature의 재무적, 비재무적 요소 판단


