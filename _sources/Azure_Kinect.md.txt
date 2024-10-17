# Azure Kinect

The Azure Kinect is a color/depth camera. We will use the depth data to perform sensor fusion with the Prophesee EVK4.

## Installation

The installation has been tested for Ubuntu 20.04 and 22.04. While it is not officially supported, a quick hack makes everything work. We developed the sensor fusion framework to run with the Kinect standalone pipeline and the ROS Wrapper. ROS is a middleware of communication that facilitates data streaming and processing. You can install ROS depending on your operating system [here](https://wiki.ros.org/ROS/Installation). Here we will use ROS Noetic as ROS2 isn't officially supported by the Azure Kinect ROS wrapper.
There is a conflict between the Prophesee camera and the Kinect, so we run the ros kinect on ubuntu 20.04 and the EVK4 on ubuntu 22.04. However it's possible to install the kinect on Ubuntu 22.04 without the ROS support.

### Ubuntu 20.04

```
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
curl -sSL https://packages.microsoft.com/config/ubuntu/18.04/prod.list | sudo tee /etc/apt/sources.list.d/microsoft-prod.list
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-get update
sudo apt install libk4a1.3-dev
sudo apt install libk4abt1.0-dev
sudo apt install k4a-tools=1.3.0
```
Clone the official Azure Kinect repository and copy the udev rules so the camera can start correctly:
```
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
cd Azure-Kinect-Sensor-SDK/
sudo cp 99-k4a.rules /etc/udev/rules.d/
```
You can check that the camera is working by running the command (reboot the computer if necessary) :
```
k4aviewer
```
Then install the ROS Wrapper inside your ROS workspace :
```
cd "your_workspace/src"
git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver
cd ..
catkin_make
```

### Ubuntu 22.04

```
sudo apt-get update
sudo apt install -y libgl1-mesa-dev libsoundio-dev libvulkan-dev libx11-dev libxcursor-dev libxinerama-dev libxrandr-dev libusb-1.0-0-dev libssl-dev libudev-dev mesa-common-dev uuid-dev
pip install numpy opencv-python
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc
sudo apt-add-repository https://packages.microsoft.com/ubuntu/20.04/prod\n
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
curl -sSL https://packages.microsoft.com/config/ubuntu/18.04/prod.list | sudo tee /etc/apt/sources.list.d/microsoft-prod.list
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt update
sudo apt-get install -y libk4a1.4
sudo apt install -y libk4a1.4-dev
wget mirrors.kernel.org/ubuntu/pool/universe/libs/libsoundio/libsoundio1_1.1.0-1_amd64.deb
sudo dpkg -i libsoundio1_1.1.0-1_amd64.deb
sudo apt install -y k4a-tools
```
Clone the Azure Kinect SDK to copy the udev rules :
```
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
cd Azure-Kinect-Sensor-SDK/
sudo cp 99-k4a.rules /etc/udev/rules.d/
```

## Runing in standalone
You can check if the kinect works properly :
```
k4aviewer
```
Then you need to select the ID of the kinect in the list, open the device then start the device. You should see the different sensors in action.

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/kinect_test.gif?raw=true)

## Runing through ROS
You only need to go through your workspace and start the launchfile :
```
cd "your_ros_workspace"
catkin_make  #if you didn't compile yet
source dev/setup.bash #if your shell if bash
roslaunch azure_kinect_ROS_Driver driver.launch
```
A ROS node starting the kinect and processing the data should be started.