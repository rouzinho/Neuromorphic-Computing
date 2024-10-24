# Kinect Loihi 2

This tutorial will show how to use the kinect interface in standalone in order to run depth images on Loihi.

## Kinect interface

The code to run this example is available inside the src [folder](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/kinect). Before using it, remember to copy the kinect_reader folder inside in order to use the process. The sample file can be found among the other samples.

```
lif_default_config = {
    "vth": 100,
    "du": 550,
    "dv": 550
}

kinect_default_config = {
    "depth_shape": (576,640),
    "filename": "/home/altair/Postdoc/Codes/kinect/simple.mkv",
    "mode": "smooth",
    "min_depth": 720,
    "max_depth": 728,
    "resize": (118,142)
}
```
First we define the parameters of the LIF neural network where the images will be send. Then, we define the kinect parameters (resolution,filename,mode) and the depth distance we wish to cover. In this example, we chose to only display to object (to some extent since the kinect frame is not aligned with the table).

```
class Architecture:
   def __init__(self,num_steps: int,use_loihi2: bool) -> None:
      self.time_steps = num_steps
      self.use_loihi2 = use_loihi2
      self.kinect_input_config = kinect_default_config
      self.lif_config = lif_default_config
      self.scaled_shape = self.kinect_input_config["resize"]

      if self.use_loihi2:
         pass
      else:
         self.run_cfg = Loihi2SimCfg(exception_proc_model_map={Dense: PyDenseModelBitAcc,LIF: PyLifModelBitAcc,KinectReader: LoihiDensePyKinectReader},select_tag="fixed_pt")

      self._create_processes()
      self._connect_processes()

   def _create_processes(self) -> None:
      self.camera = KinectReader(**self.kinect_input_config)
      self.shape_grid = self.scaled_shape
      self.lif_inp = LIF(shape=self.scaled_shape, **self.lif_config)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      self.camera.depth_out_port.connect(self.lif_inp.a_in)
      self.lif_inp.s_out.connect(self.py_receiver.a_in)

   def run(self):
      # Run
      run_cnd = RunSteps(num_steps=self.time_steps)
      self.camera.run(condition=run_cnd, run_cfg=self.run_cfg)
```

Here, we only need to create two simple processes and connect the KinectReader with the LIF. The result of this code is a smooth transformation of the depth image and only use distance between 720mm and 728mm.

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/smooth_object.gif?raw=true)

If we change the mode to "transform" and keep the same min-max values, we remove the fisheye distortion of the image :

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/transformed_object.gif?raw=true)

By changing the depth values, it is possible to select both object and robot spiking :

![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/transform_object_robot.gif?raw=true)