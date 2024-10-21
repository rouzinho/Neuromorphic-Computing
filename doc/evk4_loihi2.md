# EVK4 Loihi 2

This tutorial shows how to use the prophesee interface on Loihi

## Running Events on Loihi

First, you need to unzip all the archives in the samples [repository](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/samples). These are the events captured by the EVK4. Then you need to copy the [dvs folder](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/dvs) inside the [prophesee_loihi folder](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/prophesee_loihi). The complete code that runs the events into the loihi can be found inside the prophesee_loihi folder. In details :
```
parameters for the EVK4
prophesee_input_default_config = {
    "filename": "rect_1ms_cd.dat",
    "transformations": Compose(
        [
            Downsample(factor=0.15),
            MergePolarities(),
        ]
    ),
    "sensor_shape": (720, 1280), #480,640 for handswipe.dat because the sample comes from earlier camera version.
    "num_output_time_bins": 1,
    "sync_time": False,
    "flatten": True
}

#Parameters LIF
lif_default_config = {
    "vth": 150,
    "du": 450,
    "dv": 700
}
```
We defined the parameters for the camera by providing the filename of the datas (change the name to point towards the file you want in the samples folder), a set of transformations such as a resize and merging of event polarities (ON/OFF). Then, we define parameters for the LIF neural network.

```
class Architecture:
   def __init__(self,num_steps: int,use_loihi2: bool) -> None:
      # EVENT_RECORDING_FILENAME = "/home/altair/Postdoc/Codes/dnf_input/test.raw"
      self.time_steps = num_steps
      self.use_loihi2 = use_loihi2
      self.prophesee_input_config = prophesee_input_default_config
      self.lif_config = lif_default_config
      self.weights_config = weights_default_config
      if self.use_loihi2:
         self.run_cfg = Loihi2HwCfg(exception_proc_model_map=
                                 {PropheseeCamera: PyPropheseeCameraRawReaderModel})
      else:
         self.run_cfg = Loihi2SimCfg(
               exception_proc_model_map={Dense: PyDenseModelBitAcc,LIF: PyLifModelBitAcc,PropheseeCamera: PyPropheseeCameraRawReaderModel})
         
      self._create_processes()
      self._connect_processes()
```

Here, we declare a new class that will run the simulation. The self.run_cfg variable indicates which models are going to be used for the Dense weights connections and for the prophesee interface. Then, we create and connect the processes together.

```
   def _create_processes(self) -> None:
      self.camera = PropheseeCamera(**self.prophesee_input_config)
      self.scaled_shape = (self.camera.shape[2],self.camera.shape[3])
      self.shape_grid = self.scaled_shape
      self.lif_inp = LIF(shape=self.scaled_shape, **self.lif_config)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      self.camera.s_out.reshape(self.scaled_shape).connect(self.lif_inp.a_in)
      self.lif_inp.s_out.connect(self.py_receiver.a_in)

   def run(self):
      # Run
      run_cnd = RunSteps(num_steps=self.time_steps)
      self.camera.run(condition=run_cnd, run_cfg=self.run_cfg)
```
We begin by creating the camera process, defining the new output shape after resize and declare the LIF network. The connect function links the camera process to the LIF network. Then, we listen to the LIF network to be able to plot the datas. We did not include the weights connections to simplify the understanding of the code. Indeed, we can just lower the threshold value (vth) inside the LIF network to be able to see the neurons spiking.

The original data were a hand swiping in front of the camera and a rectangle LEGO piece moving at high speed:

Hand Swipe            |  Rectangle Moving
:-------------------------:|:-------------------------:
![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/hand_swipe.gif?raw=true)  |  ![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/rect_prophesee.gif?raw=true) 