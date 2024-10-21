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
from lava.lib.dnf.kernels.kernels import Kernel
from lava.lib.dnf.kernels.kernels import MultiPeakKernel, SelectiveKernel
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights, Convolution
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

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
    
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    

prophesee_input_default_config = {
    "filename": "rect_1ms_cd.dat",
    "transformations": Compose(
        [
            Downsample(factor=0.1),
            MergePolarities(),
        ]
    ),
    "sensor_shape": (720, 1280),
    "num_output_time_bins": 1,
    "sync_time": False,
    "flatten": True
}


lif_default_config = {
    "vth": 250,
    "du": 350,
    "dv": 600
}

weights_default_config = {
    "w_ff": 255,
    "w_rec": 5,
    "w_o": 16,
    "kernel_size": 20
}

multipeak_dnf_default_config = {
    "in_conn": {
        "weight": 8
    },
    "lif": {"du": 2000,
            "dv": 2000,
            "vth": 25

            },
    "rec_conn": {
        "amp_exc": 48,
        "width_exc": [20, 20],
        "amp_inh": -35,
        "width_inh": [39, 39]
    },
    "out_conn": {
        "weight": 100,
    }
}

selective_dnf_default_config = {
    "lif": {"du": 2009,
            "dv": 2047,
            "vth": 30
            },
    "rec_conn": {
        "amp_exc": 7,
        "width_exc": [10, 10],
        "global_inh": -5
    }
}

class Architecture:
   def __init__(self,num_steps: int,use_loihi2: bool,kernel: Kernel) -> None:
      # EVENT_RECORDING_FILENAME = "/home/altair/Postdoc/Codes/dnf_input/test.raw"
      self.time_steps = num_steps
      self.use_loihi2 = use_loihi2
      self.prophesee_input_config = prophesee_input_default_config
      self.lif_config = lif_default_config
      self.weights_config = weights_default_config
      self.multipeak_dnf_config = multipeak_dnf_default_config
      selective_dnf_config = selective_dnf_default_config
      self.kernel = kernel
      self.multipeak_in_params = self.multipeak_dnf_config["in_conn"]
      self.multipeak_lif_params = self.multipeak_dnf_config["lif"]
      self.multipeak_rec_params = self.multipeak_dnf_config["rec_conn"]
      self.multipeak_out_params = self.multipeak_dnf_config["out_conn"]
      self.selective_lif_params = selective_dnf_config["lif"]
      self.selective_rec_params = selective_dnf_config["rec_conn"]
      if self.use_loihi2:
         self.run_cfg = Loihi2HwCfg(exception_proc_model_map=
                                 {PropheseeCamera: PyPropheseeCameraRawReaderModel})
      else:
         self.run_cfg = Loihi2SimCfg(
               exception_proc_model_map={PropheseeCamera: PyPropheseeCameraRawReaderModel},select_tag="fixed_pt")
         
      self._create_processes()
      self._connect_processes()
      
   def _create_processes(self) -> None:
      self.camera = PropheseeCamera(**self.prophesee_input_config)
      self.scaled_shape = (self.camera.shape[2],self.camera.shape[3])
      self.shape_grid = self.scaled_shape
      #self.lif_inp = LIF(shape=self.scaled_shape, **self.lif_config)
      self.dnf_multipeak = LIF(shape=self.scaled_shape,**self.multipeak_lif_params)
      self.dnf_selective = LIF(shape=self.scaled_shape,**self.selective_lif_params)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      connect(self.camera.s_out.reshape(self.scaled_shape), self.dnf_multipeak.a_in,
                 ops=[Weights(**self.multipeak_in_params)])
      
      #multipeaks
      connect(self.dnf_multipeak.s_out, self.dnf_multipeak.a_in,
               ops=[Convolution(MultiPeakKernel(**self.multipeak_rec_params))])
      
      #selective
      #connect(self.dnf_multipeak.s_out, self.dnf_selective.a_in,
      #          ops=[Weights(**self.multipeak_out_params)])

        # Connections around selective dnf
      #connect(self.dnf_selective.s_out, self.dnf_selective.a_in,
      #         ops=[Convolution(SelectiveKernel(**self.selective_rec_params))])

      #self.dnf_selective.s_out.connect(self.py_receiver.a_in)
      self.dnf_multipeak.s_out.connect(self.py_receiver.a_in)

   def run(self):
      # Run
      run_cnd = RunSteps(num_steps=self.time_steps)
      self.camera.run(condition=run_cnd, run_cfg=self.run_cfg)

   def plot(self):
      # Get probed data from monitors
      self.data_dnf = self.py_receiver.data.get().transpose()
      self.data_dnf = np.transpose(self.data_dnf,(0,2,1))
      print(self.data_dnf.shape)
      # Stop the execution of the network
      self.camera.stop()
      # Generate an animated plot from the probed data
      self.grid = np.ones(self.shape_grid)
      self.fig = plt.figure()
      self.im = plt.imshow(self.grid, vmin=0.0, vmax=1.0,animated=True)
      anim = matplotlib.animation.FuncAnimation(self.fig, self.update_gauss, interval=5, frames=self.time_steps, blit=True)
      #plt.show()
      anim.save('/home/altair/Postdoc/multi_dnf_big.mp4', writer = 'ffmpeg', fps = 30)

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
   detection_kernel = MultiPeakKernel(amp_exc=83, width_exc=3.75, amp_inh=-70, width_inh=7.5)
   architecture = Architecture(120,False,detection_kernel)
   architecture.run()
   architecture.plot()
   
   
   
   