Recent Models on Pose Estimation
===

1. References
  - Pose Estimation: [https://paperswithcode.com/task/pose-estimation](https://paperswithcode.com/task/pose-estimation)
  - Multi-Person Pose Estimation: [https://paperswithcode.com/task/multi-person-pose-estimation](https://paperswithcode.com/task/multi-person-pose-estimation)
  - Keypoint Detection: [https://paperswithcode.com/task/keypoint-detection](https://paperswithcode.com/task/keypoint-detection)

2. Recently published, but no codes (21, Nov, 2019)
  - Cascade Feature Aggregation (2019)
  - DarkPose (2019, repository exsists but not ready)
  - Spacial Contextual Information (2019)

3. To understand pose estimation models, we need to know sub tasks of pose estimation. Mostly these are mixed, but some of models focused on specific task.
  - Single / Multi person pose estimation
  - Keypoint detection (COCO Keypoint Detection Challenge)
  - Pose tracking (PoseTrack Challenge)
  - Refinement pose estimation

AlphaPose(RMPE's New Version) (2016)
---
**"RMPE: Regional Multi-person Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1612.00137v5)
[[Code Link]](https://github.com/MVIG-SJTU/AlphaPose)
<br/>
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

SSTN은 STN과 SDTN(Spatial De-Transformer Network)으로 구성되며, SPPE의 전후에 추가된다. STN은 human detector로부터 region을 입력받아 해당 영역에서 사람을 이미지 가운데 위치시키도록 훈련받는다 (STN을 훈련할땐 Parallel SPPE를 이용한다고 함). 
STN의 transform 결과는 사람의 위치, 회전, 크기를 변형시킨 것이므로 SPPE를 통해 포즈를 계산한뒤, 포즈가 원본 영역에서 제대로된 위치를 찾기 위해 SDTN을 수행한다. 결과적으로 SSTN은 SPPE가 human detector로부터 제안된 영역에서 포즈 추정을 할때 최대의 성능을 끌어내도록 돕는 역할을 한다.


Cascaded Pyramid Network(CPN+) (2017)
---
**"Cascaded Pyramid Network for Multi-Person Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1711.07319v2)
[[Code Link]](https://github.com/chenyilun95/tf-cpn)

<p align="center"><img src="./images/CPN.png"></p>


SimplePose(ResNet) (2018)
---
**"Simple Baselines for Human Pose Estimation and Tracking"**

[[Paper Link]](https://arxiv.org/abs/1804.06208v2)
[[Code Link]](https://github.com/Microsoft/human-pose-estimation.pytorch)

<p align="center"><img src="./images/ResNet.png" width="90%"></p>

<p align="center"><img src="./images/ResNet2.png" width="90%"></p>


HRNet (2019)
---
**"Deep High-Resolution Representation Learning for Human Pose Estimation"**

[[Paper Link]](https://arxiv.org/abs/1902.09212v1)
[[Code Link]](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

<p align="center"><img src="./images/HRNet.jpg" width="90%"></p>