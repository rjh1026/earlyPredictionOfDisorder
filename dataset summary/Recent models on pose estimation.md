Recent Models on Pose Estimation
===

1. References
    - Pose Estimation: [https://paperswithcode.com/task/pose-estimation](https://paperswithcode.com/task/pose-estimation)
    - Multi-Person Pose Estimation: [https://paperswithcode.com/task/multi-person-pose-estimation](https://paperswithcode.com/task/multi-person-pose-estimation)
    - Keypoint Detection: [https://paperswithcode.com/task/keypoint-detection](https://paperswithcode.com/task/keypoint-detection)

2. Recently published, but no codes (as 21, Nov, 2019)
    - Cascade Feature Aggregation (2019)
    - DarkPose (2019, repository exsists but not ready)
    - Spacial Contextual Information (2019)

3. To understand pose estimation models, we need to know sub tasks of pose estimation. Mostly these are mixed, but some of models focused on specific task.
    - Single / Multi person pose estimation
    - Keypoint detection (COCO Keypoint Detection Challenge)
    - Pose tracking (PoseTrack Challenge)
    - Refinement pose estimation
	
4. In pose estimation pipe lines, there is two approaches. 
    1. Top-Down Approaches(two-step framework): firstly locate and crop all persons from images, and then solve the single person pose estimation problem in the cropped person patches.
    2. Bottom-Up Approaches(part-based framework): directly predict all keypoints at first and assemble them into full poses of all persons.

RMPE (2016)
---
**"RMPE: Regional Multi-person Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1612.00137v5)
[[Code Link]](https://github.com/MVIG-SJTU/AlphaPose) * Code link is connected to AlphaPose that is new version of RMPE.

Ref: [STN (Spatial Transformer Network)](https://jamiekang.github.io/2017/05/27/spatial-transformer-networks/)

<p align="center"><img src="./images/RMPE.PNG"></p>
<br/>
<p align="center"><img src="./images/RMPE2.PNG"></p>

- SPPE: Single Person Pose Estimator
- SSTN(STN+SDTN): Symmetric Spatial Transformer Network
- p-Pose NMS: Parametric Pose Non-Maximum-Suppression
- PGPG: Pose-Guided Proposals Generator

<br/>

RMPE는 기본적으로 human detector를 통해 region proposal을 수행한 뒤, 각각의 region에서 SPPE를 적용해 pose를 추출해내는 two-step framework 이다. human detector와 SPPE는 기존의 다양한 방법들을 사용할 수 있고, 실제 테스트로 faster-rcnn (human detector)과 stacked hourglass model (SPPE)를 사용하였다.

RMPE는 human detector가 올바르지 못한 region을 제안하더라도 pose estimation 하기에 무리가 없도록 하는 것을 목표로 한다.
잘못된 region으로부터 localization error와 redundant detection problem이 발생하는데, 이를 SSTN과 p-Pose NMS를 통해 극복하였다. 

SSTN은 STN과 SDTN(Spatial De-Transformer Network)으로 구성되며, SPPE의 전후에 추가된다. 
STN은 이미지 속 객체를 transform 시켜 객체를 효과적으로 파악할 수 있도록 해준다 (STN 링크 참고).
STN은 human detector로부터 region을 입력받게 되는데, 해당 영역에서 사람을 이미지 가운데 위치시키도록 훈련시킨다 (STN을 훈련할땐 Parallel SPPE를 이용). 이는 SPPE가 더 효과적으로 사람객체를 파악할 수 있도록 해준다. 

STN의 transform 결과는 사람의 위치, 회전, 크기를 변형시킨 것이다. 따라서 STN의 결과를 SPPE에 전달하여 포즈를 계산한 뒤, 다시 포즈가 원본 영역에서 제대로된 위치를 찾도록 SDTN을 수행한다. 
결과적으로 SSTN은 SPPE가 human detector로부터 제안된 영역에서 포즈 추정을 할때 최대의 성능을 끌어내도록 돕는 역할을 한다.


Cascaded Pyramid Network(CPN+) (2017)
---
**"Cascaded Pyramid Network for Multi-Person Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1711.07319v2)
[[Code Link]](https://github.com/chenyilun95/tf-cpn)

Ref: [[ResNet]](http://openresearch.ai/t/resnet-deep-residual-learning-for-image-recognition/41) [[RoIAlign of Mask R-CNN]](https://cdm98.tistory.com/33) [[FPN]](https://eehoeskrap.tistory.com/300) 

<p align="center"><img src="./images/CPN.png"></p>
<p align="center"><img src="./images/CPN2.png"></p>

- simple keypoints: eyes, hands that easly recognizable
- hard keypoints: occluded and invisible keypoints, complex background

<br/>

CPN은 simple keypoints를 찾는 것뿐만이 아니라 hard keypoints 문제까지도 해결하고자 하였다. CPN은 two stages 구조로 접근하여 GlobalNet과 RefineNet으로 구성된다. 
GlobalNet은 feature pyramid net(FPN)을 기반으로 하여 simple keypoints를 찾는다. 
RefineNet은 GlobalNet으로부터 모든 레벨의 feature representations을 통합함으로써 hard keypoints를 다룬다. 

CPN의 GlobalNet 네트워크 구조는 ResNet backbone에 기반한다. ResNet의 conv features conv2~5의 마지막 residual blocks에 대해 3x3 컨볼루션 필터를 적용하여 heatmap을 생성한다. 이러한 heatmap들의 spatial resolution과 semantic information은 서로 대립적인 관계가 성립됨을 주목해, 이를 둘다 유지하기 위한 FPN의 U-shape 구조를 사용한다.

그러나 GlobalNet 단일 네트워크로는 hard keypoints를 탐지하기에 부족하므로, RefineNet으로 upsampling과 concatenating을 하여 레벨 간 정보를 통합시킨다 (HyperNet 방식). 


Simple Baseline Model (2018)
---
**"Simple Baselines for Human Pose Estimation and Tracking"**

[[Paper Link]](https://arxiv.org/abs/1804.06208v2)
[[Code Link]](https://github.com/Microsoft/human-pose-estimation.pytorch)

Ref: [[Deconvolution]](https://dambaekday.tistory.com/3) [[Batch Normalization]](https://light-tree.tistory.com/139)

<p align="center"><img src="./images/DHN.png" width="70%"></p>

<br/>

이 모델은 이전의 복잡한 모델들(Hourglass, CPN, etc) 보다 상당히 간단한 구조로 약간 더 높은 정확도를 보여 주목을 받게 되었다. 연구의 방향은 '복잡한 것과 반대로 간단한 구조가 얼마나 좋은 성능을 보일수 있는가'를 나타내는 것에 목적을 두고 있다. 

DHN은 ResNet을 기반으로, ResNet의 마지막 컨볼루션 stage에 deconvolutional layers를 추가하였다. 3개의 deconvolutional layer와 batch normalization 그리고 ReLU가 사용된다. 각 레이어는 4x4 커널을 가진 256 필터를 사용한다. 마지막은 k개의 keypoint를 나타내는 heatmaps을 생성하기 위해 1x1 convolutional layer가 사용된다. 

기존의 모델들과의 중요한 차이점은 high resolution feature map을 생성하기 위해 upsampling, put convolutional parameters 를 따로 사용하지 않고, 이 두 방법을 skip layer connection 없이 deconvolutional layer로 통합시킨다.


HRNet (2019)
---
**"Deep High-Resolution Representation Learning for Human Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1902.09212v1)
[[Code Link]](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

<p align="center"><img src="./images/HRNet.jpg" width="90%"></p>


OccNet, OccNetCB (2019)
---
**"Human Pose Estimation for Real-World Crowded Scenarios"**

[[Paper Link]](https://arxiv.org/abs/1907.06922)
[[Code Link]](https://github.com/thomasgolda/Human-Pose-Estimation-for-Real-World-Crowded-Scenarios)

<p align="center"><img src="./images/OccNet.PNG" width="70%"></p>
<p align="center"><img src="./images/OccNet2.PNG"></p>
<p align="center"><img src="./images/OccNet3.PNG"></p>

<br/>

해당 논문에서는 군중들에 대한 포즈 추정과 같은 (감시카메라를 통한 군중 행동 감지 등) 복잡한 환경에는 몇가지 해결해야할 과제가 있음을 말하고 있다. 공공장소에서 감시 카메라로부터 얻는 영상은 partially occluded people, mutual occlusions by humans, low resolution of persons와 같이 센서의 제한 혹은 카메라로부터의 거리 때문에 발생하는 문제가 있으므로 포즈 추정이 어려움을 주목한다. 따라서 이들은 사람들로 복잡한 감시 시나리오 환경을 구성하는 방법 및 데이터셋을 소개하며, occluded keypoints까지 탐지할 수 있는 모델을 제시한다. 

이들은 군중 데이터셋으로 유명한 CrowdPose 데이터셋과 JTA(Joint Track Auto)라는 데이터셋을 비교하여 보완하였다. CrowdPose는 실세계의 사람들을 촬영한 이미지 데이터셋이기에 포즈가 다양하여 keypoint distribution이 넓게 퍼져있지만 대부분 카메라를 향해 바라보고 있다는 단점이 있다. 반면 JTA는 GTA게임 속 가상 환경에서 촬영된 이미지 데이터셋이므로 대부분 걸어다니는 자세라서 distribution이 좁지만, 사람이 카메라를 향하는 경우와 향하지 않는 경우 둘다 충분히 많다는 장점이 있다.
이러한 차이를 고려해 JTA 데이터셋의 단점을 보완한 (모드를 사용하여 sitting, yoga, push-ups 등 동작 추가) JTA-Ext 데이터셋을 소개하며 3가지 모두 훈련과 실험에 사용하였다. 

이들이 제시한 모델은 기존의 simple baseline model에서 occluded keypoints 탐지를 위한 branch를 추가함으로써 확장하였다. occluded and visble keypoints를 위한 branch가 서로 연결되어 있지만 서로 다른 task를 맡기는 이러한 구조는 occlusion을 다루는데 강한 모델을 만들어준다고 말하고 있다. OccNet은 backbone으로 ResNet50을 사용하며 simple baseline model과 마찬가지로 top-down 접근법을 사용한다. OccNet은 2개의 transposed convolution (deconvolution, up-convolution 등) 뒤에 나누어지는 반면, OccNetCB(Cross Branch)는 1개의 transposed convolution 뒤에 나뉘도록 하였다. 

CPM (2016)
---
**"Convolutional Pose Machine"**

[[Paper Link]](https://arxiv.org/abs/1602.00134)
[[Code Link]](https://github.com/shihenw/convolutional-pose-machines-release)

Ref: [[Class Activation Map (Heat Map)]](https://kangbk0120.github.io/articles/2018-02/cam)

<p align="center"><img src="http://openresearch.ai/uploads/default/original/1X/e50f409aa672c5eed54dc83d325a3cec6de98f3a.jpg"></p>
<p align="center"><img src="http://openresearch.ai/uploads/default/original/1X/18f20fd72702bd2157277019938753e79bc6b5fb.jpg"></p>

<br/>

CPM은 CVPR'16에서 발표되어 최근 주목받고 있는 Part Affinity Fields (OpenPose) 연구의 기초가 되었다.

CPM은 sequential prediction 구조를 통해 predictor가 점차 정교하게 예측하도록 유도한다. 이전 stage의 predictor가 생성한 belief map (heat map)과 현 stage의 image feature를 통해 더 나은 예측을 보여주는 belief map을 생성하게 된다. 여기서 predictor는 입력 이미지의 특정 위치에서 예측하려는 part 마다 localization score를 계산한다.

CPM의 특징은 receptive field가 좁은 영역에서 점차 넓은 영역을 다룰 수 있도록 크기를 증가시켜 많은 spatial context information을 포함시킨다는 점이다. 이러한 large receptive field는 모델로 하여금 멀리 떨어진 parts 간의 관계까지 학습하도록 돕는다. 인식하기 쉬운 part의 정보가 애매해서 인식하기 어려운 part의 예측에 도움을 준다고 말하고 있다.

이렇게 깊은 컨볼루션 레이어를 쌓다보면 vanishing gradients 문제가 발생하는데, 이를 각 stage마다 loss layer를 추가하는 방식으로 개선하였다.
