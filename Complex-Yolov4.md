## Complex-Yolov4

> *3D Object Detection을 위한 모델
> 3D data : Point Cloud 데이터를 사용*
##### 특징
- one-stage YOLO 응용한 것 (one-stage는 물체가 있을 만한 region 예측 없이 전체 이미지로 바로 예측)
- E-RPN 사용함으로써 다른 모델에 비해 모든 class를 한번에 분류

##### 데이터 pipeline
1. RGB-Map (일반적으로 생각하는 RGB 사진) 으로 제작
2. CNN
3. E-RPN Grid
![[Screenshot 2024-09-12 at 12.43.28 AM.png]]

###### Bird's-eye View (조감도, 이하 BEV)
- Lidar 3D 데이터를 XY projection 해서 2D로 표현한 프레임(= z값이 0)

###### Point Cloud Pre-processing
- 3D 데이터를 BEV RGB Map으로 변환하는 작업
- 아래 수식에서 $P$는 3D 공간의 좌표값
- x∈[0,40m],y∈[−40m,40m],z∈[−2m,1.25m] 의미는, - x축 기준으로 0m에서 40m 사이의 범위, y축 기준 -40m에서 40m 기준 공간을 의미
- $P_{Ω→j}​$ 은 3D 공간의 좌표를 새로운 공간 (ex. 2D)로 투영된 점 (위 범위를 몇 개의 픽셀로 나누는지 결정하면 RGB-Map 의 픽셀값이 정해질 수 있음)
- $S_j$ 은 Grid Map을 RGB 값으로 나타낸 점들의 부분집합
- $z_g$ maximum height, $z_b$ maximum intensity, $z_r$ Nromalized density 는 $S_j$ 에서 최대 높이, 밀도 등 속성 정보 뽑아내는 것
![[Screenshot 2024-09-12 at 12.48.51 AM.png]]

##### Architecture
- Complex 단어는 E-RPN Grid를 추가했기 때문에 붙인 이름
- 모델이 물체의 방위 값 (방향이라 생각)을 포함하는 것이 큰 특징
- CNN 모델은 YOLOv4의 특징을 그대로 가짐
![[Screenshot 2024-09-12 at 1.07.38 AM.png]]

##### Euler-Region-Proposal (E-RPN)
- 3D 상에 위치하는 객체 표현을 위해 complex angle 값을 추가
- complex angle은 위치, 크기, orientation 정보
- orientation은 방향과 유사

###### Anchor Box
*미리 정의된 크기와 모양을 가진 box를 움직여 모델이 각 grid cell에서 실제 객체가 box 와 얼마나 일치하는지 판단*
- 3D object detection에서 prior은 크기와 방향이 설정된 anchor box라고 생각하면 됨
- 기존 object detection과 다르게 방위각을 측정하고 있음
- 5개의 box를 가지는 것으로 합의 
- 2 angle directions, 3 다른 사이즈 box
	- **차량 크기(heading up)**: 차량이 위쪽을 향하고 있는 경우의 prior 상자
	- **차량 크기(heading down)**: 차량이 아래쪽을 향하고 있는 경우의 prior 상자
	- **자전거 크기(heading up)**: 자전거가 위쪽을 향하는 경우의 prior 상자
	- **자전거 크기(heading down)**: 자전거가 아래쪽을 향하는 경우의 prior 상자
	- **보행자 크기(heading left)**: 보행자가 왼쪽을 향하는 경우의 prior 상자

<기존 2D에서의 anchor box 사용 용도>
![[Screenshot 2024-09-12 at 1.22.42 AM.png]]
![[Screenshot 2024-09-12 at 1.21.51 AM.png]]

##### Loss Function
$L = L_{Yolo} + L_{Euler}$
$L_{Yolo}$
-> 객체를 인식하지 않아도 되는 grid에 대한 연산:
- $λ_{noobj}$ 는 객체가 없는 grid cell에 대해 굳이 객체를 불필요한 예측 x
-  $λ_{noobj}$ 값에 0 값을 대입함으로써 연산 항을 날려버림
- $1^{\text{noobj}}_{ij}$ 객체가 존재하지 않는 grid cell에 활성화 (no object)
- 따라서 객체가 없는 cell에 대해서 예측 안하도록 Loss function 값을 줄이고자 노력해야 함

-> 객체를 검출해야 하는 grid에 대한 연산:
- 객체가 검출되어야 하는 cell에 대해서는 $1^{\text{obj}}_{ij}$ 항이 활성화되어 해당 grid cell에 대한 객체의 좌표, 크기, 클래스 확률을 예측

$L_{Euler}$
-> 방위각 $\phi$ 에 대한 loss 값 계산 

##### 성능 비교
- KITTI 데이터셋 기준으로 Complex YOLOv4 모델 성능 가장 높음
- 성능 측정 지표는 mAP, FPS 두가지로 했음
- mAP 는 다양한 객체를 얼마나 정확하게 탐지하는 지에 대한 정확도 지표
- FPS 초당 처리할 수 있는 프레임 수를 의미하며 실시간 성능 지표
- mAP, FPS 매우 높은 값을 가짐
![[Screenshot 2024-09-12 at 1.49.19 AM.png]]

##### 정밀도 비교 - BEV 기준
- Complex YOLOv4 모델이 실시간 성능 지표 FPS는 다른 모델들에 비해 월등히 높지만,
- 정밀도 AP 값은 Pedestrian, Cyclist 면에서 살짝 낮은 성능
- AP 값이 Car은 높다

##### 정밀도 비교 - 3D Object Detection기준
- Complex YOLOv4 모델이 실시간 성능 지표 FPS는 다른 모델들에 비해 월등히 높지만,
- 정밀도 AP 값은 Pedestrian, Cyclist 면에서 살짝 낮은 성능
- AP 값이 Car은 높다

#### 결론
Complex YOLOv4는 실시간 성능 (FPS) 중심으로 설계되었고, 차량 (Car) 탐지에 두각을 보임