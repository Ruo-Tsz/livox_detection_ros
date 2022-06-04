# livox_detection_ros
It's a package to run [LIVOX detector](https://github.com/Livox-SDK/livox_detection.git) on ros node.

## 1. build package
```
bash *PATH_TO_livox_detection_ros*/build_pkg.sh
```

## 2. launch nodes by
For all nodes works properly, we need to launch detection node and tracking node accordingly in 2 terminals.
1. launch detection node
```
roslaunch *PATH_TO_livox_detection_ros*/launch/livox_det.launch
```

## Model
Please fisrt download the model and named directory as **model**, then put it under **livox_detection** directory. <br />
[model link](https://drive.google.com/file/d/1hSAqYCL3WJOqfiNrsGHw-RaWrSnzDlj8/view?usp=sharing)
