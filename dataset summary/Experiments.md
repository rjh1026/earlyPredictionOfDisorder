Datasets
===
SSBD
---

In SSBD dataset, they provide xml file with each video like as below.
One video has multiple behaviours (armflapping, headbanging, spinning), so we can split it into clips by the behaviour-detected time.

```xml
<video id="v_ArmFlapping_01" keyword="10_yr_severe_autistic">
   <url>http://www.youtube.com/watch?v=I7fdv1q9-m8</url>
   <height>360</height>
   <width>480</width>
   <frames>1365</frames>
   <persons>1</persons>
   <duration>46s</duration>
   <conversation>yes</conversation>
   <behaviours count="2" id="b_Set_01">
   	  <behaviour id="b_01">		
         <time>18:24</time>		
         <bodypart>hand</bodypart>		
         <category>armflapping</category>		
         <intensity>high</intensity>		
         <modality>video</modality>	
      </behaviour>
	  <behaviour id="b_02">		
         <time>27:32</time>		
         <bodypart>full</bodypart>		
         <category>headbanging</category>		
         <intensity>high</intensity>		
         <modality>video</modality>	
      </behaviour>
   </behaviours>
</video>
```
---

Models
===

OpenPose
---

~~test version: OpenPose 1.5.0~~ <br/>
~~test model: OpenPose[-COCO model](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md#body_25-vs-coco-vs-mpi-models)~~ <br/>

~~I exported keypoints from the clips using OpenPose with cpu only mode. Because of the processing time (cpu mode: 0.3 fps, gpu mode: 10~15 fps), I tested a few samples only. The result is in [./videos/openpose_ssbd](./videos/openpose_ssbd).~~

training guide: [How to train OpenPose models with COCO images](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/training#whole-body-training) **(_but there is no guide for custom training_)**

caffe model: [openpose_caffe_train](https://github.com/CMU-Perceptual-Computing-Lab/openpose_caffe_train)

### Input Format

In their training guide they use LMDB files and these are hard to see inside. I guess they followed [COCO annotation format](http://cocodataset.org/#format-data) rules. 

### Pose Order (COCO)

They followed COCO annotation format which has 17 keypoints and added one more feature(Neck)  interpolated by Lsho and Rsho.

<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png", width="480">
</p>

### Output Format

```
{
    "version":1.1,
    "people":[
        {
            "pose_keypoints_2d":[582.349,507.866,0.845918,746.975,631.307,0.587007,...],
			# The length of list will be 18 (OpenPose-COCO model).
        }
    ]
}
```

AlphaPose
---

They support training code from scratch. we can see inside of the model that implemented with pytorch also.
 
training code: [train.py](https://github.com/MVIG-SJTU/AlphaPose/blob/master/scripts/train.py)

pytorch model: [simplepose.py](https://github.com/MVIG-SJTU/AlphaPose/blob/master/alphapose/models/simplepose.py)

### Input Format

it seems to use COCO json format showed as above. 
it contains 17 keypoints. <br/>
```
['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
```

### Pose Order and Output Format

Keypoint ordering and Output format are described here [alphaPose - output format](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md#keypoint-ordering).

Simple Baseline model
---

training code: [train.py](https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/pose_estimation/train.py)

pytorch model: [pose_resnet.py](https://github.com/microsoft/human-pose-estimation.pytorch/blob/2d723e3fd7f93dd81dd093af2328174555f6d552/lib/models/pose_resnet.py)

### Input Format
same as above. 17 keypoints.

### Pose Order and Output Format
