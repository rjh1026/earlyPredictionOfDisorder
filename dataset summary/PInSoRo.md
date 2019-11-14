The Plymouth Interactive Social Robots dataset (PInSoRo)
===
[https://github.com/freeplay-sandbox/dataset](https://github.com/freeplay-sandbox/dataset)

Summary
---
The PInSoRo Dataset (also called Freeplay Sandbox dataset) is a large (120 children, 45h+ of RGB-D video recordings), open-data dataset of child-child and child-robot social interactions. <br/>
These interactions are recorded during little-constrained free play episodes. They emcompass a rich and diverse set of social behaviours.

**PInSoRo is open-data, but they only provide the anonymised version for legal and ethical reasons.** <br/>
The anonymised version has `pinsoro-*.csv` files, but no raw video streams and ROS bag files that can be modified by presented tools.<br/>
Details is here: [https://freeplay-sandbox.github.io/get-dataset](https://freeplay-sandbox.github.io/get-dataset)

The PInSoRo Dataset consists in 75 recordings of interactions (45 child-child interactions; 30 child-robot interactions; up to 40 min per interactions). And each record has `pinsoro-*.csv` file containing facial features, skeletons, gaze, etc.

Dataset Structure
---
In this section, I describe only structure of the dataset focusing on our interest what is the useful information.<br/>
Details is here: [https://github.com/freeplay-sandbox/dataset/tree/master/data](https://github.com/freeplay-sandbox/dataset/tree/master/data)

#### Meta-data (github repository) ####
<details markdown="1">
<summary> <b>experiment.yaml</b>: contains the interaction details. </summary>

```yaml
timestamp: 1496917355889909029 # timestamp of the begining of the interaction
condition: childchild # condition (childchild or childrobot)
purple-participant:
  id: 2017-06-08-11:18-p1
  age: 4
  gender: female
  details:
    tablet-familiarity: 0  # self-reported familiarity with tablets, from 0 (no familiarity to 2 (familiar)
yellow-participant: # absent in the child-robot condition
  id: 2017-06-08-11:18-y1
  age: 4
  gender: female
  details:
    tablet-familiarity: 2
markers:  # events of interest, annotated during the experiment by the experimenter. Timestamps in seconds from the begining
  75.936775922: interesting
  104.153236866: interesting
  214.65380907: interesting
  328.371172904: interesting
  376.429432868: interesting
  428.393737077: interesting
  590.867943048: issue
  685.981807947: interesting
  708.350763082: issue
  789.571500062: interesting
  811.970171928: interesting
notes: # open-ended notes, taken by the experimenter during the experiment. Timestamp in seconds from the begining.
  general: Both very quiet. P has done experiment before (1y002).
  75: Very quiet
  104: Y watching P
  214: Both absorbed in own games
  328: Confusion about colours
  376: P drawing pictures
  428: Quiet battle about colours
  590: P to FS "Look!"
  685: Y copied P's drawing
  708: P seeking encouragement from FS
  780: P drawing pictures, Y scribbling
  811: Both seem kind of bored
postprocess: # (optional) details of specific post-processing performed on this recording
    - recompressed sandtray background, start timestamp moved from 1496917354.451842 to 1498168785.467365
issues: # (optional) specific issues with this recording
    - skeleton extraction failed
```
</details>


#### Anonymised Dataset ####

The data is sampled at 30Hz. It starts at the first video frame of either of the 2 cameras filming the children's faces. 

Note that the children were wearing brightly colored sport bibs (the left child had a yellow one, the right child a purple one). The (left) camera filming the (right) purple child is accordingly refered to as the purple camera, and the (right) camera filming the yellow child as the yellow camera.

Besides, in the child-robot condition, the robot was always replacing the yellow child. Hence, in that condition, all yellow child-related data is missing.

<details markdown="1">
<summary> <b>pinsoro-*.csv</b>: All the main dataset features sampled at 30Hz (30FPS). </summary>

This file contains 449 fields. But I Summed up these to a few fields needed to us.

##### CSV Fields #####
- `condition`: child-child or child-robot. [Refer to the
  website](https://freeplay-sandbox.github.io/dataset) for details.

- `purple_child_age`, `purple_child_gender`, `yellow_child_age`, `yellow_child_gender`: self explanatory

- `purple_frame_idx`: index of the frame in the purple camera video stream.
  Can be used to quickly extract a specific frame or range of frame in the video stream.

- `purple_child_face{00..69}_{x,y}`: 2D coordinates of the 70 facial landmarks
  (including pupils), normalised in [0.0, 1.0], extracted by
  [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/). See
  [OpenPose documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#face-output-format)
  for the location of these landmarks.
<p align="center"><img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_face.png" width="80%" height="80%"></p>

- `purple_child_skel{00..17}_{x,y}`: 2D coordinates of the 18 skeleton
  keypoints, normalised in [0.0,1.0], extracted by
  [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/). See
  [OpenPose documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-coco)
  for the location of these keypoints. Note that, due to the experimental
  setting generating a lot of occlusion (children sitting in front of a table),
  the skeletal data is not always reliable.
<p align="center"><img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png/" width="250" height="400"></p>

- `purple_child_head_{x,y,z,rx,ry,rz`: head pose estimation, in m and rad, relative to the table centre (see below for the camera extrinsics). Computed using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).
<p align="center"><a href="https://www.youtube.com/watch?v=V7rV0uy7heQ" target="_blank"><img src="http://img.youtube.com/vi/V7rV0uy7heQ/0.jpg" alt="Multiple Face Tracking" width="240" height="180" border="10" /></a></p>

- `purple_child_gaze_{x,y,z}`: gaze vector, averaged for both eyes, relative to the table centre. Computed using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).
<p align="center"><img src="https://github.com/TadasBaltrusaitis/OpenFace/raw/master/imgs/gaze_ex.png" width="70%"></p>

- `purple_child_au{01,02,04,05,06,07,09,10,12,14,15,17,20,23,25,26,28,45}`:
  Intensity of 18
  facial action units, extract using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). See [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) for the details.
<p align="center"><img src="https://raw.githubusercontent.com/wiki/TadasBaltrusaitis/OpenFace/images/AUs.jpg" width="70%"></p>

- `purple_child_motion_intensity_{avg,stdev,max}`: average, standard deviation
  and maximum of the magnitude of the motion observed in the frame. This is
  computed by performing a [optical flow computation using the Dual TVL1 algorithm](https://github.com/freeplay-sandbox/analysis/blob/master/src/optical_flow.cpp#L163) and averaging the resulting values on the whole frame

- `purple_child_motion_direction_{avg,stdev}`: average and standard deviation
  of the direction of the motion observed in the frame. This is
  computed by performing a [optical flow computation using the Dual TVL1 algorithm](https://github.com/freeplay-sandbox/analysis/blob/master/src/optical_flow.cpp#L163) and averaging the resulting values on the whole frame.
<p align="center"><img src="http://amroamroamro.github.io/mexopencv/opencv/tvl1_optical_flow_demo_03.png" witdh="500px" height="400px"></p>

- `{purple,yellow}_child_{task_engagement,social_engagement,social_attitude}`: manual annotations of the social interaction. See the [coding scheme.](https://freeplay-sandbox.github.io/coding-scheme). If more that one annotator annotated this frame, **and the annotators disagreed**, the different annotations are separated by a `+`.
<p align="center"><img width="500px" height="300px" src="https://freeplay-sandbox.github.io/media/coding-scheme.png"></p>

</details>


<details markdown="1">
<summary> <b>freeplay.poses.json</b>: stores the skeletons and facial features extracted from each of the video frames.</summary>

##### Format of poses files #####

```
 {<topic_name>:
    {"frames" : [{
        "ts": <timestamp in floating sec>,
        "poses": {
            "1": [ # pose index (purple child)
                # x,y in image coordinates (pixels), c is confidence in [0.0,1.0]
                [x, y, c], # 0- Nose
                [x, y, c], # 1- Neck
                [x, y, c], # 2- RShoulder
                [x, y, c], # 3- RElbow
                [x, y, c], # 4- RWrist
                [x, y, c], # 5- LShoulder
                [x, y, c], # 6- LElbow
                [x, y, c], # 7- LWrist
                [x, y, c], # 8- RHip
                [x, y, c], # 9- RKnee
                [x, y, c], # 10- RAnkle
                [x, y, c], # 11- LHip
                [x, y, c], # 12- LKnee
                [x, y, c], # 13- LAnkle
                [x, y, c], # 14- REye
                [x, y, c], # 15- LEye
                [x, y, c], # 16- REar
                [x, y, c]  # 17- LEar
            ],
            "2": [ # if present (yellow child), second skeleton
              ...
            ]
      },
      "faces": {
            "1": [ # face index (purple child)
              # x,y in image coordinates, c is confidence in [0.0,1.0]
              [x, y, c],
              ... # 70 points in total, see OpenPose website for indices
            ],
            "2": [ # (yellow child)
               ...
            ]
      }
      "hands": {
            "1": { # hand index
                "left": [
                    # x,y in image coordinates, c is confidence in [0.0,1.0]
                    [x, y, c],
                    ... # 20 points in total, see OpenPose website for indices
                ],
                "right": [
                    # x,y in image coordinates, c is confidence in [0.0,1.0]
                    [x, y, c],
                    ... # 20 points in total
                ]
            },
            "2":
              ...
        }
    },
    { # 2nd frame
        "ts": ...
        "poses":
        ...
    }
    ]
  }
}
```

Because these JSON files are typically large (>100MB for 20-25 min), they
recommend us carefully choose our [JSON
library](https://github.com/miloyip/nativejson-benchmark) both in terms of
parsing speed and memory requirements.

</details>

Presented Tools
---
There are two tools of Data Analysis and Annotation, but we can't use of them without full dataset version.