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

from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.proc.dense.process import Dense
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from lava.magma.core.resources import CPU
from lava.proc.lif.process import LIF
from lava.proc.lif.models import PyLifModelBitAcc
from lava.proc.io.sink import RingBuffer
from lava.utils.system import Loihi2
from IPython.display import HTML
from IPython import display
from bags_reader.process import BagReader, LoihiDensePyBagReader
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter

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

   def plot(self):
      # Get probed data from monitors
      self.data_dnf = self.py_receiver.data.get().transpose()
      print(self.data_dnf.shape)
      # Stop the execution of the network
      self.camera.stop()
      # Generate an animated plot from the probed data
      self.grid = np.ones(self.shape_grid)
      self.fig = plt.figure()
      self.im = plt.imshow(self.grid, vmin=0.0, vmax=1.0,animated=True)
      anim = matplotlib.animation.FuncAnimation(self.fig, self.update_gauss, interval=50, frames=self.time_steps, blit=True)
      plt.show()
      #anim.save('/home/altair/Postdoc/Codes/frame_loihi/depth_rect_static.mp4', writer = 'ffmpeg', fps = 30)

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

if __name__ == '__main__':
   architecture = Architecture(70,False)
   architecture.run()
   architecture.plot()
   #plt.show()