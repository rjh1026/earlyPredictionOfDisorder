Before to start
===

모델 훈련 과정을 직접 실행해보기 위해선 실험 환경을 준비해야합니다. 
본 실험을 진행한 간략한 스펙은 다음과 같습니다.
- OS: Window 10 (64bit)
- CPU: Intel Core i7 9700
- GPU: NVIDIA GeForce GTX 1080 Ti (GPU Memory: 11GBytes)
- Memory: 32GBytes

또한, 주요 모듈 버전은 다음과 같습니다.
- **python 3.6, pytorch 1.5.0 for cudatoolkit=10.2**
- CUDA 10.2, cuDNN 7.6.5 for CUDA 10.2 (버전은 사용하는 GPU에 따라 달라질 수 있으나, 10.0 이상 권장)

훈련에 앞서 반드시 GPU 메모리가 충분한지 확인하기 바랍니다.<br/> 코드를 직접 실행을 할 수 없다면 `.ipynb` 파일에 코드와 실행결과를 같이 기록했으니 참고하세요.<br/>
`.ipynb` 파일은 github에서도 직접 볼 수 있지만 간혹 렌더링되지 않는 경우가 있기 때문에, [https://nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)사이트를 이용하시는 것을 추천합니다. 

Contents
===

학습 과정은 간단한 CNN(Convolutional Neural Network) 모델부터 Pose Estimation 연구 분야에 좋은 성과를 낸 복잡한 모델들까지 순차적으로 살펴보는 것에 초점을 맞추었습니다.

1. [실험 환경 설치](./1_setting_environment.ipynb)
2. [파이토치 기초](./2_pytorch_basic.ipynb)
3. [CNN](./3_convolutional_neural_network.ipynb)
4. [포즈 데이터셋과 데이터 전처리](./4_pose_dataset_and_preprocessing.ipynb)
5. [포즈 추정 모델](./5_pose_estimation_models.ipynb)
6. [모델 성능 검증과 결과 시각화](./6_evaluation_and_visualization.ipynb)