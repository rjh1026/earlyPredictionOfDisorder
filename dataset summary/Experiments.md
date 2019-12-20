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

version: OpenPose 1.5.0 <br/>
model: OpenPose [COCO model](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/quick_start.md#body_25-vs-coco-vs-mpi-models) <br/>

I exported keypoints from the clips using OpenPose with cpu only mode. Because of the processing time (cpu mode: 0.3 fps, gpu mode: 10~15 fps), I tested a few samples only. The result is in [./videos/openpose_ssbd](./videos/openpose_ssbd).
