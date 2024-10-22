# ROS Loihi 2

This tutorial will show how to play a rosbag of depth images to Loihi.

## The interface

We are going to create a python class that uses the ROS interface :

```
lif_default_config = {
    "vth": 20,
    "du": 550,
    "dv": 550
}

rosbag_test_config = {
    "depth_shape": (642,1204),
    "filename": "/home/altair/Postdoc/Codes/frame_loihi/test_moving.bag",
    "topic": "/depth_perception/depth",
    "resize": (64,120)
}
```
Here, we define the parameters for the LIF neural network that will run the spikes from the rosbag. Then we define the parameters of the ros interface. First, we need to setup the image size resolution (1204x642) and the filename of the rosbag (inside the samples [folder](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/samples)). Then, we should indicate the name of the topic we recorded and the final resolution that will run on the Loihi.

```
class Architecture:
   def __init__(self,num_steps: int,use_loihi2: bool) -> None:
      self.time_steps = num_steps
      self.use_loihi2 = use_loihi2
      self.bag_input_config = rosbag_test_config
      self.lif_config = lif_default_config

      if self.use_loihi2:
         pass
      else:
         self.run_cfg = Loihi2SimCfg(exception_proc_model_map={Dense: PyDenseModelBitAcc,LIF: PyLifModelBitAcc,BagReader: LoihiDensePyBagReader},select_tag="fixed_pt")

      self._create_processes()
      self._connect_processes()

   def _create_processes(self) -> None:
      self.camera = BagReader(**self.bag_input_config)
      self.scaled_shape = (64,120)
      self.shape_grid = self.scaled_shape
      self.lif_inp = LIF(shape=self.scaled_shape, **self.lif_config)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      connect(self.camera.depth_out_port, self.lif_inp.a_in,
                 ops=[Weights(10)])
      self.lif_inp.s_out.connect(self.py_receiver.a_in)

   def run(self):
      # Run
      run_cnd = RunSteps(num_steps=self.time_steps)
      self.camera.run(condition=run_cnd, run_cfg=self.run_cfg)
```
As for the EVK4, we need to create the processes that consists of a BagReader and a LIF network. Then, we connect the output of the BagReader to the input of the LIF. You can find the complete code . Do not forget to copy the folder bags_reader inside the LoihiBag to be able to use the rosbag interface. we run the code with 2 rosbags : one with a static LEGO piece and one with the LEGO moving.

LEGO static            |  LEGO Moving
:-------------------------:|:-------------------------:
![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/depth_rect_static.gif?raw=true)  |  ![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/depth_rect_moving.gif?raw=true) 

