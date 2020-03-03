Datasets
===

SSBD
---

SSBD 데이터셋은 3가지 종류의 행동 (Arm flapping, Headbanging, Spinning)을 보이는 유튜브 영상들로 이루어져 있다. 유튜브 사용자들이 공개한 영상들이기 때문에 SSBD에서 제공하는 일부 영상들은 비공개되거나 삭제되어 사용할 수 없다. 또한, 다운받을 수 있는 영상들 중에도 상태가 좋지 않은 영상들 (서로 다른 resolution, 심한 노이즈, 어두움, 정지된 영상, 끊김 등)이 있다는 것을 감안한다.

실험을 위해 다음과 같은 과정을 수행하였다.
  1. 데이터셋이 제공하는 annotation 파일(`.xml`)을 기준으로, 각 원본 영상에서 (e.g. v_ArmFlapping_01.avi) 두드러진 Action이 관찰된 구간만 참조한다.
  2. 참조한 구간을 [오픈소스 툴](https://github.com/antran89/clipping_ssbd_videos)을 사용해 ???x320 size의 영상으로 추출한다.

- ArmFlapping
<p align="center"> <img src="../images/ssbd_armflapping.gif"> </p>
<br/>

- HeadBanging
<p align="center"> <img src="../images/ssbd_headbanging.gif"> </p>
<br/>

- Spinning
<p align="center"> <img src="../images/ssbd_spinning.gif"> </p>
<br/>

Infant Normative Dataset
---

생후 20주차 이내의 건강한 아이들의 움직임을 녹화한 유튜브 영상들 (100개 이상)로 구성되어 있다. 영상들은 대체로 양호하지만, 촬영할 때의 각도나 영상 속 아이의 머리가 향한 방향이 조금씩 다르다 (e.g. 오른쪽 방향으로 누워있는 아이, 위쪽 방향으로 누워 있는 아이).

실험을 위해 다음과 같은 과정을 수행하였다.
  1. 데이터셋이 제공하는 annotation 파일(`.csv`)을 기준으로, 각 원본 영상에서 (e.g. 4.avi) 아이의 모습만 촬영된 구간을 참조한다.
  2. 참조한 구간을 MoviePy를 사용해 ???x320 size의 영상으로 추출한다.

- example 1
<p align="center"> <img src="../images/infant1.gif"> </p>
<br/>

- example 2
<p align="center"> <img src="../images/infant2.gif"> </p>
<br/>

---

Experiments
===

Comparison
---
