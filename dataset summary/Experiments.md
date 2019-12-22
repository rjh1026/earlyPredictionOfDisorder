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

test version: OpenPose 1.5.0 <br/>
test model: OpenPose[-COCO model](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md#body_25-vs-coco-vs-mpi-models) <br/>

I exported keypoints from the clips using OpenPose with cpu only mode. Because of the processing time (cpu mode: 0.3 fps, gpu mode: 10~15 fps), I tested a few samples only. The result is in [./videos/openpose_ssbd](./videos/openpose_ssbd).

They describes [How to train OpenPose models with COCO images](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/training#whole-body-training), **but there is no guide for custom training.**

### Input Format

In their caffe model training guide they use LMDB files and these are hard to see inside. But I guess they followed [COCO annotation format](http://cocodataset.org/#format-data) rules. 

### Pose Output (COCO)

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

They support training code from scratch. we can see inside of the model that implemented with pytorch also. [AlphaPose train.py](https://github.com/MVIG-SJTU/AlphaPose/blob/master/scripts/train.py)
, [How to train issue](https://github.com/MVIG-SJTU/AlphaPose/issues/62)

### Input Format

### Output Format

Output format and default keypoint ordering are described here [Keypoint Ordering](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md#keypoint-ordering).

