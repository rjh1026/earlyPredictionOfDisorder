Well-known Datasets In Pose Estimation
===

<details markdown="1">
<summary> <b>2D Pose Estimation Dataset</b> </summary>

MPII Human Pose Dataset
---
[http://human-pose.mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de)

<p align="center"> <img src="http://human-pose.mpi-inf.mpg.de/images/random_activities.png"> </p>

- Around 25K images containing over 40K people with annotated body joints.
- 410 human activities with activity labels.
- Each image was extracted from a YouTube video and provided with preceding and following un-annotated frames.
- Annotations are stored in a matlab structure.
- Automatic evaluation server that evaluates differently depending on single/multi-person. 
- Performance analysis tools based on rich test set annotations that doesn't provide to tester.

Leeds Sports Poses
---
[https://sam.johnson.io/research/lsp.html](https://sam.johnson.io/research/lsp.html)

<p align="center"> <img src="https://sam.johnson.io/research/images/dataset/bas2.jpg"> </p>

- 2000 pose annotated images of mostly sports people gathered from Flickr.
- 14 joint locations with MATLAB format. (2x{ankle, knee, hip, wrist, elbow, shoulder}, Neck, Head top)
- The images have been scaled such that the most prominent person is roughly 150 pixels in length.
- Left and right joints are consistently labelled from a person-centric viewpoint.

FLIC(Frames Labelled In Cinema)
---
[https://bensapp.github.io/flic-dataset.html](https://bensapp.github.io/flic-dataset.html)
<p align="center"> <img src="https://jonathantompson.github.io/flic_plus_files/sample.jpg"> </p>

- 5003 images from popular Hollywood movies.
- 10 upperbody joints with MATLAB format.
- no occlusion or severely non-frontal.

COCO 2017 Keypoint Detection dataset
---
In case of Keypoint Detection of COCO, 2017 Keypoint dataset is the latest version.

<p align="center"> <img src="http://cocodataset.org/images/keypoints-splash.png"> </p>

- train/val/test images containing more than 200K images and 250K person instances.
- Annotated with [JSON format](http://cocodataset.org/#format-data). Each keypoint has location x, y and a visibility flag v (v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible).

VGG Human Pose Estimation Datasets
---
[https://www.robots.ox.ac.uk/~vgg/data/pose/index.html](https://www.robots.ox.ac.uk/~vgg/data/pose/index.html)

VGG contains several datasets as below.

### Youtube Pose ###
<p align="center"> <img src="https://www.robots.ox.ac.uk/~vgg/data/pose/array.png"> </p>

- 50 YouTube videos for upper body pose estimation.
- 2D locations of upper body joints with MATLAB format.

### BBC Pose (Default, Extended, Short) ###
<p align="center"> <img src="https://www.robots.ox.ac.uk/~vgg/data/pose/img/examplevid.png"> </p>

- 20 videos (in default case) recorded from BBC with an overlaid sign language interpreter.
- annotated with upper body joints (head, wrists, elbows, shoulders)
- split into train/validation/test sets (10/5/5)

Pose Track 2018
---
[https://posetrack.net/users/download.php](https://posetrack.net/users/download.php)
<br/>
Reference: [Posetrack Data set: Summary](https://medium.com/@anuj_shah/posetrack-data-set-summary-9cf61fc6f44e)


<p align="center"> <img src="https://miro.medium.com/max/800/1*D7hq1ULqzLedjkhwH1NwPg.jpeg"> </p>

Posetrack is a new large-scale benchmark for video-based human pose estimation and articulated tracking.

This dataset focuses on 3 task.<br/>
1) Single-frame multi-person pose estimation<br/>
2) Multi-person pose estimation in videos<br/>
3) Multi-person articulated tracking

- 1356 video sequences, 46K annotated frames, 276K body pose annotations.
- Annotation stored in [JSON format](https://github.com/leonid-pishchulin/poseval)
- Training/Validation/Test sets
- PoseTrack17: 15 keypoints, PoseTrack18: 17 keypoints

Self-Stimulatory Behaviours in the Wild for Autism Diagnosis Dataset (SSBD)
---
[https://rolandgoecke.net/research/datasets/ssbd/](https://rolandgoecke.net/research/datasets/ssbd/)<br/>
[https://github.com/antran89/clipping_ssbd_videos](https://github.com/antran89/clipping_ssbd_videos)

<p align="center"><img src="./images/Flapping_his_arms.gif"></p>

- extremely challenging (low quality, occulusion, etc) for analysis as they are recorded in uncontrolled natural settings.
- 75 URLs of videos with average duration of 90 sec.
- 3 group (arm flapping, head banging, spinning)
- XML annotation format.


</details>


<details markdown="1">
<summary> <b>3D Pose Estimation Dataset</b> </summary>

ITOP
---
[https://www.alberthaque.com/projects/viewpoint_3d_pose/#dataset](https://www.alberthaque.com/projects/viewpoint_3d_pose/#dataset)
<p align="center"><img src="https://www.alberthaque.com/projects/viewpoint_3d_pose/img/front.jpg"> <img src="https://www.alberthaque.com/projects/viewpoint_3d_pose/img/top_labeled.jpg"></p>

- 100K annotated depth images from extreme viewpoints (top, front-view).
- 15 joints
- depth map, point cloud

DensePose-COCO, PoseTrack
---
[http://densepose.org](http://densepose.org)

<p align="center"><img src="http://densepose.org/img/anno/anno2.png"></p>

DensePose aims at mapping all human pixels of an RGB image to the 3D surface of the human body (They call this correspondence). So its datasets contain correspondences only fit on their methods.

The Moving Infants In RGB-D dataset (MINI-RGBD)
---
[https://www.iosb.fraunhofer.de/servlet/is/82920/](https://www.iosb.fraunhofer.de/servlet/is/82920/)
<p align="center"><img src="https://www.iosb.fraunhofer.de/servlet/is/82920/acvr2018_sample.png"></p>

MINI-RGBD dataset is focused on generating infant 2D/3D pose dataset without privacy problems. To preserve the privacy of infants, they generated new textures and shapes by averaging multiple texture and shapes of real infants using [SMIL](https://www.youtube.com/watch?v=aahF1xGurmM&feature=youtu.be) (Multi-Infant Linear body model).

<p align="center"><img src="https://www.iosb.fraunhofer.de/servlet/is/82920/smil_sample.png"></p>

**This dataset wasn't opened at the web, but they offer dataset only for requests.**

(Following features are not certain.)

- 2D and 3D ground truths (skeletons)
- infants up to the age of 7 months
- continuous motions instead of single frames
- RGB, depth images

</details>