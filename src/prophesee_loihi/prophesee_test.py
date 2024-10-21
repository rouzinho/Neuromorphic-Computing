#!/usr/bin/env python3

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import warnings
import locale
#import pandas
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
warnings.filterwarnings("ignore")

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from dvs.prophesee import PropheseeCamera,PyPropheseeCameraEventsIteratorModel, PyPropheseeCameraRawReaderModel
from dvs.transformation import Compose, Downsample, MergePolarities
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.monitor.process import Monitor
from lava.proc.sparse.process import Sparse
from lava.proc.dense.process import Dense
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.magma.core.resources import CPU
from lava.proc.lif.process import LIF
from lava.proc.lif.models import PyLifModelBitAcc
from lava.proc.sparse.models import PySparseModelBitAcc
from metavision_sdk_cv import TrailFilterAlgorithm, ActivityNoiseFilterAlgorithm
from utils import EventSpike
from lava.proc.io.sink import RingBuffer
import metavision_core as mv
from lava.utils.system import Loihi2
from IPython.display import HTML
from IPython import display
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

#Here we check if the code is running on CPU or Loihi
if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
    
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    
#parameters for the EVK4
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

weights_default_config = {
    "w_ff": 255,
    "w_rec": 5,
    "w_o": 16,
    "kernel_size": 20
}

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

   def plot(self):
      # Get probed data from monitors
      self.data_dnf = self.py_receiver.data.get().transpose()
      self.data_dnf = np.transpose(self.data_dnf,(0,2,1))
      # Stop the execution of the network
      self.camera.stop()
      # Generate an animated plot from the probed data
      self.grid = np.ones(self.shape_grid)
      self.fig = plt.figure()
      self.im = plt.imshow(self.grid, vmin=0.0, vmax=1.0,animated=True)
      anim = matplotlib.animation.FuncAnimation(self.fig, self.update_gauss, interval=20, frames=self.time_steps, blit=True)
      plt.show()
      #anim.save('/home/altair/Postdoc/Codes/neurocam/hand_swipe.mp4', writer = 'ffmpeg', fps = 30)

   def setup_plot_gauss(self):
      self.im.set_data(self.grid)
      self.ax0.axis([0, self.shape_grid[0], 0.0, self.shape_grid[1]])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.im, 

   def update_gauss(self, i):
      self.grid = self.data_dnf[i]
      self.im.set_data(self.grid)
      print(i)

      return self.im, 

   def animated_plot(self):
      pass

if __name__ == '__main__':
   architecture = Architecture(200,False)
   architecture.run()
   architecture.plot()
   
   
   
   