Datasets
===

SSBD
---

SSBD 데이터셋은 3가지 종류의 행동 양상(Arm flapping, Headbanging, Spinning)을 띄는 유튜브 영상들로 이루어져 있다. 유튜브 사용자들이 공개한 영상들이기 때문에 현재 일부 영상들은 비공개되거나 삭제되어 찾을 수 없다. 또한, 다운받을 수 있는 영상들 중에도 상태가 좋지 않은 영상들(서로 다른 resolution, 심한 노이즈, 어두움, 정지된 영상, 끊김 등)이 있다는 것을 감안한다.

실험을 위해 다음과 같은 과정을 수행하였다.
  1. SSBD 데이터셋이 제공하는 annotation 파일(`.xml`)을 기준으로, 각 원본 영상에서 (e.g. v_ArmFlapping_01.avi) 두드러진 Action이 관찰된 구간만 (최대 150frame) 참조한다.
  2. 참조한 구간을 [오픈소스 툴](https://github.com/antran89/clipping_ssbd_videos)을 사용해 320x240 fixed size의 새로운 영상으로 추출한다.

추출한 영상들은 [./videos/ssbd_clip_segment/](./videos/ssbd_clip_segment/) 에서 확인할 수 있다. (일부 깨지거나 왜곡된 영상들이 존재함)

---

Output Format
---

### Pose Order

실험에 사용한 모델들의 output format은 COCO 17 keypoints로 통일시킨다.

<p align="center">
    <img src="http://www.programmersought.com/images/935/c3a73bf51c47252f4a33566327e30a87.png", width="480">
</p>

<br/>

### Result
```
['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
```

```
{
    "people":[
        {
            "pose_keypoints_2d":[582.349,507.866,0.845918,746.975,631.307,0.587007,...],
			# The length of list will be 18 (OpenPose-COCO model).
        }
    ]
}
```
