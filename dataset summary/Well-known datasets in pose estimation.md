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

COCO 2019 Keypoint Detection
---
...

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

DensePose-COCO
---
...

DensePose-PoseTrack
---
...

</details>