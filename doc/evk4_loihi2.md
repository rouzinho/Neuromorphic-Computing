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

The results of these events running on Loihi are the following :

Hand Swipe Loihi            |  Rectangle Moving Loihi
:-------------------------:|:-------------------------:
![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/hand_swipe_loihi.gif?raw=true)  |  ![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/rect_opt_loihi.gif?raw=true) 


## Loihi with DNFs

It is possible to apply some parameters to the LIF in order to make them behave like a dynamic neural field. For a selective field, add this code at the begining :

```
multipeak_dnf_default_config = {
    "in_conn": {
        "weight": 8
    },
    "lif": {"du": 2000,
            "dv": 2000,
            "vth": 25

            },
    "rec_conn": {
        "amp_exc": 14,
        "width_exc": [5, 5],
        "amp_inh": -10,
        "width_inh": [9, 9]
    },
    "out_conn": {
        "weight": 100,
    }
}

selective_dnf_default_config = {
    "lif": {"du": 2009,
            "dv": 2047,
            "vth": 15
            },
    "rec_conn": {
        "amp_exc": 7,
        "width_exc": [15, 15],
        "global_inh": -5
    }
}
```

It will define the parameters of the selective and multipeak activation kernel. Then connect them :

```
def _create_processes(self) -> None:
      self.camera = PropheseeCamera(**self.prophesee_input_config)
      self.scaled_shape = (self.camera.shape[2],self.camera.shape[3])
      self.shape_grid = self.scaled_shape
      #self.lif_inp = LIF(shape=self.scaled_shape, **self.lif_config)
      self.dnf_multipeak = LIF(shape=self.scaled_shape,**self.multipeak_lif_params)
      self.dnf_selective = LIF(shape=self.scaled_shape,**self.selective_lif_params)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      #No real multipeaks, only the weights value
      connect(self.camera.s_out.reshape(self.scaled_shape), self.dnf_multipeak.a_in,
                 ops=[Weights(**self.multipeak_in_params)])

      
      #selective
      connect(self.dnf_multipeak.s_out, self.dnf_selective.a_in,
                ops=[Weights(**self.multipeak_out_params)])

        # Connections around selective dnf
      connect(self.dnf_selective.s_out, self.dnf_selective.a_in,
               ops=[Convolution(SelectiveKernel(**self.selective_rec_params))])

      self.dnf_selective.s_out.connect(self.py_receiver.a_in)
```

 The selective field will rpovide this result with the LEGO toy :
 ![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/selective_dnf.gif?raw=true)

 If we apply a multipeak kernel with specific parameters, it can create a working memory by keeping a trail of the events :

 ![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/multi_dnf.gif?raw=true)

 With different parameters, the traces can be more sparse :

 ![test](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/multi_dnf_big.gif?raw=true)