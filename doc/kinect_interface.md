# Kinect Interface

We will introduce how to record samples from the kinect and explain the process that read depth images from the camera and send them as inputs to the Loihi chip 

## Kinect recording

In order to use the process, you first need to install the pykinect_azure library :
```
#source your virtualenv
source neuro/bin/activate
pip install pykinect_azure
```

Then you can record a sample by runing this command :
```
k4arecorder -d WFOV_UNBINNED -c 1080p -r 15 -l 10 wfov.mkv
```
WFOV_UNBINNED is the option to choose the wide angle of the depth camera. The option -r 15 means that we record at 15fps and -l 10 indicates the length in seconds of the recording. More information on this command can be found [here](https://learn.microsoft.com/bg-bg/previous-versions/azure/kinect-dk/azure-kinect-recorder)

## Kinect process

The class can take several parameters such as the size of the depth (1024x1024 for WFOV option). More information about camera parameters are available [here](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/namespace_microsoft_1_1_azure_1_1_kinect_1_1_sensor_ae1bee72789a1fe39e868e5b39ba62108.html). The mode parameter indicate how you want to process the depth images : depth, smooth depth, tranformed (transform depth to color frame - removing the fisheye distortion). Here is an example :
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/smoothdepth.gif?raw=true)
![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/transformed.gif?raw=true)

The other parameters are min_depth and max_depth, which defines a depth you want to monitor. Everything outside this interval will by excluded from the depth image.
```
import sys
import numpy as np
import time
from threading import Thread
import cv2
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import pykinect_azure as pykinect
import ctypes
from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.image import Image

class KinectReader(AbstractProcess):
   """
   Process that read depth images from ros bag file and send them through outport 

   Parameters
   ----------
   depth_shape: tuple
      shape of the depth image.
   file_name: str
      name of the recording.
   mode: str
      type of depth image. "depth","smooth","transformed"
   min_depth: int
      minimal depth to process.
   max_depth: int
      maximal depth to process.
   resize: tuple
      resizing the images if the parameter is set.
   """
   def __init__(
            self,
            depth_shape: tuple,
            filename: str = "",
            mode: str = "depth",
            min_depth: int = 0,
            max_depth: int = 10000,
            resize: tuple = None) -> None:
      if filename == "":
         raise ValueError(
            "you must provide the path of a bag"
         )
      if mode != "":
         self.mode = mode
      self.filename = filename
      self.min_depth = min_depth
      self.max_depth = max_depth
      self.resize = resize
      if resize is not None:
         self.depth_shape = self.resize
      else:
         self.depth_shape = depth_shape
      self.depth_out_port = OutPort(shape=self.depth_shape)
      super().__init__(
         depth_shape=self.depth_shape,
         filename=self.filename,
         mode=self.mode,
         min_depth=self.min_depth,
         max_depth=self.max_depth,
         resize=self.resize
      )

#class implementing the bag reader
@implements(proc=KinectReader, protocol=LoihiProtocol)
@requires(CPU)
class LoihiDensePyKinectReader(PyLoihiProcessModel):
   depth_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

   def __init__(self, proc_params: dict) -> None:
      super().__init__(proc_params)
      self._file_name = proc_params["filename"] 
      self._depth_shape = proc_params["depth_shape"]
      self._mode = proc_params["mode"]
      self._min_depth = proc_params["min_depth"]
      self._max_depth = proc_params["max_depth"]
      self._resize = proc_params["resize"]
      pykinect.initialize_libraries()
      self._playback = pykinect.start_playback(self._file_name)
      self._playback_config = self._playback.get_record_configuration()

   def run_spk(self):
      #print("get image")
      depth_image = self._get_kinect_frame()
      self.depth_out_port.send(depth_image)

   def _get_kinect_frame(self):
      #get next frame
      ret, capture = self._playback.update()
      depth_image = None
      cv_image_resized = None
      #setup mode
      if ret:
         if self._mode == "depth":
            ret_depth, depth_image = capture.get_depth_image()
         if self._mode == "smooth":
            ret_depth, depth_image = capture.get_smooth_depth_image()
         if self._mode == "transformed":
            ret_depth, depth_image = capture.get_transformed_depth_image()
         #resize to param size
         cv_image_resized = cv2.resize(depth_image, (self._resize[1],self._resize[0]), interpolation = cv2.INTER_AREA)
         #exclude depth values not between min_depth and max_depth
         for i in range(cv_image_resized.shape[0]):
            for j in range(cv_image_resized.shape[1]):
               if cv_image_resized[i,j] > self._max_depth or cv_image_resized[i,j] < self._min_depth:
                  cv_image_resized[i,j] = 0
      else:
         cv_image_resized = np.zeros((self._resize))

      return cv_image_resized
   
   def _pause(self):
      """Pause was called by the runtime"""
      super()._pause()
      self.t_pause = time.time_ns()

   def _stop(self) -> None:
      """Stop was called by the runtime"""
      super()._stop()   
```

The use of this process will be demonstrated in the tutorial section.

