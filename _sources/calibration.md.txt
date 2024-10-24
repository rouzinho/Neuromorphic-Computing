# Calibration

This page is dedicated to the calibration of both cameras. The goal is to align the depth camera and the DVS on the same frame with an homography. 

## Robot/Kinect calibration

### Extrinsic parameters

We will calibrate the depth camera by using ROS and specific packages we developed. First, we introduce the robot and its setup. The robot used is the PincherX150 from [Trossen robotics](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/px150.html). You will find the installation and setup on the website. In our case, we will use the Kinect camera instead of the Intel D435 used by the manufacturer. 
In the xsarm_perception.launch, replace the line 
```
 <arg name="cloud_topic"                       default="/camera/depth/color/points"/>
```
by 
```
 <arg name="cloud_topic"                       default="/points2"/>
```
which is the topic where the azure kinect is publishing the pointcloud. Then we need to make some modifications inside the ros kinect wrapper. Open the driver.launch file inside the azure_kinect_ros_wrapper package and do the following changes :
```
 <arg name="cloud_topic"                       default="/points2"/>
```
This essentially setup the depth mode to a narrower field of view.
Then you can begin the robot camera calibration by starting the kinect. Don't forget to setup the April TAG as explained in [the documentation](https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros1_packages/perception_pipeline_configuration.html) . In a terminal, go to your ROS workspace and type :
```
 <arg name="cloud_topic"                       default="/points2"/>
```
Open an other terminal and type :
```
roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=px150 use_pointcloud_tuner_gui:=true use_armtag_tuner_gui:=true
```
This will open the configuration setup and you can follow the instruction from the documentation.
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/robot_settings.png?raw=true)

Once your robot arm is calibrated with the camera and that the extrinsic calibration between them seems accurate, you can proceed.

### Depthmap calibration

We developed a simple tool that generate a depthmap based on the pointcloud delivered by the Kinect. It is a ROS package that will find the right parameters in order to project the 3D points on a surface through a simple linear transformation. First, copy the package visualize_depth present in the src folder of this documentation and copy it inside your ROS workspace :
```
cp -R Neuromorphic-Computing/src/visualize_depth "your_ros_workspace/src"
cd "your_ros_workspace"
catkin_make
```
Now modify the launch file inside the visualize_depth package :
```
<param name="init_params" type="bool" value="false" />
```
This means that the package will print the right parameters for your environment at start. Then, can start the package :
```
roslaunch visualize_depth visualize_depth.launch
```
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/visualize_depth.png?raw=true)
You can stop the program (Ctrl-C) after a few seconds once the values are printed inside the terminal. Then, insert these values inside the launch file and set the init_params to false. From now on, your depth camera is calibrated to your environment. You can use the package visualize_frame to verify that an object is in the frame.

## Computing the Homography

The idea is to select several points in the depth frame and find the corresponding points within the DVS frame. However, the only way to vizualize a static object on the prophesee camera is to use a blinking LED. A handcraft blinking LED (100Hz) with an Arduino is easy to setup. Then, we have to define several location on the scene. Use the visualize_depth package and metastudio to make sure the locations you chose are visible in both frame.

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/scene.jpg?raw=true)

To gather points in the depthmap, we are going to use the package calibration_hom. Copy the package inside your ROS workspace :
```
cp -R Neuromorphic-Computing/src/calibration_hom "your_ros_workspace/src"
cd "your_ros_workspace"
catkin_make
```
Then copy the depthmap parameters from the visualize_depth launch file to the calibration_hom launchfile (e.g calibrate_depth.launch). You should also indicate the path where the points are going to be saved. The package locate the depth of the LED then save these coordinates inside a file.

To gather points within the DVS camera, you can use the folder homography_dvs and more particularly the python code calibration.py. This script detect the blinking led and save the coordinates inside a text file.

To resume, the ROS package calibration_hom detect the LED in the depthmap and save the coordinates inside a txt file. The script calibration.py does the same for the DVS in a different txt file. Both txt files will contains the coordinates of the LED in the respective frames.

The protocol is simple. First Place the blinking LED at the first location :

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/led.jpg?raw=true)

Start the calibration_hom package :
```
roslaunch calibration_hom calibrate_depth.launch
```
You should see the depth of the LED and the red dot represent the saved point :
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/screen_calib_depth.png?raw=true)

You can stop the program and start the calibration.py

```
python3 calibration.py
```
You should see the blinking LED from the DVS for a few second, then the program stop and the position of the point is saved in the txt file.
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/calib_dvs.gif?raw=true)

You can now move the LED to a new location and start again. For an accurate calibration, 8-9 points are necessary. You should place the 2 text files containings the points inside the homography_dvs folder.
Once you have done that, you can start the script homography.py inside the homography_dvs folder. 

```
python3 homography.py
```

The program will compute the homography transform from the depth camera to the DVS and output a YAML file that would perform the transformation of the frame later on. This file will be used by the ROS kinect interface to transform the depth data before using them as input to the Neuromorphic chip Loihi 2.