# Autonomous_Driving_Challenge
[Team 꼬부기] 주행환경의 차량/버스를 인식하고 해당 객체의 의미론적(Semantic) 위치와 후미등 상태를 인식하는 동시에 이미지 개체 분할(Instance Segmentation)을 진행하는 챌린지입니다.
![image](https://github.com/user-attachments/assets/72332663-92a8-4814-aca9-71ac5afcd6a3)

## Competition
대회 소개: [ 2024년 자율주행 인공지능 챌린지](https://www.aiotkorea.or.kr/2024/webzine/KIoT/2024%EC%9E%90%EC%9C%A8%EC%A3%BC%ED%96%89AI%EC%B1%8C%EB%A6%B0%EC%A7%80%EC%B0%B8%EC%97%AC%EA%B0%80%EC%9D%B4%EB%93%9C%EB%9D%BC%EC%9D%B8.pdf)  
최종 순위: [6/20]

## Data
### Dataset Split

## EDA

## Model
YOLOv8를 변형하여 사용
```python
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv10 object detection model. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nagent: 2 #number of total
nloc: 5 #number of locations
nact: 4 #number of actions
nc: [2,5,4] # number of classes

scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  s: [0.33, 0.50, 1024]
  l: [1.00, 1.00, 512]


backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Multi_v10Segment, [nc, 32, 256]] # Detect(P3, P4, P5)
```
Multi_v10Segment 모듈을 사용해 classification과 segmentation을 동시에 수행한다.
객체의 종류, 위치, 상태 각각에 대해 독립적인 예측이 가능하다.

## Attempted Methods & Results
### 성능 향상 ⬆️
| 기법 | 설명 | 
|------|------|
| **모델 Scale 변경 (s → l)** | 모델이 더 깊고 넓은 특성을 학습 |
| **IoU Loss 함수 변경 (CIoU → WiseIoU)** | Dynamic Focusing Mechanism을 통해 바운딩 박스 회귀 과정에서 더 정교한 예측 수행 |
| **Mask Ratio 조정 (4 → 1)** | 더 많은 원본 데이터 정보를 활용 |
| **Batch 크기 변경 (32 → 4)** | 더 다양한 미니배치의 특성을 반영하여 일반화 성능 향상 |
| **추론 과정에서 conf 조정 (0.001 → 0.25)** | 불확실한 예측을 제거 |

### 성능 저하 ⬇️
| 기법 | 설명 |
|------|------|
| **cls Loss 함수 변경 (bce → focal)** | 불균형 데이터에 적합한 Loss 함수 사용 |
| **AFPN** |다양한 피처 스케일을 조합하여 특성 강화 및 다중 레이어 활용 | 
| **CARAFE** | 업샘플링 단계에서의 해상도 복원을 개선하여 디테일 보강 |
| **Gold-YOLO** | 더 효율적으로 객체를 탐지하도록 설계 |
| **GaussNoise** | 가우시안 노이즈 추가 |

## Challenges
- 대회 기간 동안 총 5번만 결과를 제출할 수 있었다.
- 데이터 불균형으로 인해 차량의 상태를 분류하는 데 있어서 좌측/우측 방향지시등을 잘 감지하지 못했다.

## Conclusion
- AFPN, GOLD-YOLO 등 모델의 복잡성을 높이는 시도들은 오히려 학습을 방해하는 것을 확인하였다.
- 모델이 학습해야 할 핵심 정보를 강화하고 오차를 줄이는 방향에서 성능이 향상되었다.
