#!/usr/bin/env python3

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
import warnings
import locale
#import pandas
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
warnings.filterwarnings("ignore")

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights
from dvs.prophesee_sync import PropheseeCameraSync,PyPropheseeCameraEventsIteratorModel, PyPropheseeCameraRawReaderModelSync
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
from lava.proc.io.sink import RingBuffer
import metavision_core as mv
from lava.utils.system import Loihi2
from IPython.display import HTML
from IPython import display
from bags_reader.process_sync import BagReaderSync, LoihiPyBagReaderDVS
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
    
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")
    

prophesee_input_default_config = {
    "filename": "/home/altair/Postdoc/Codes/samples/rect_fast_cut_cd.dat",
    "transformations": Compose(
        [
            Downsample(factor=0.1),
            MergePolarities(),
        ]
    ),
    "sensor_shape": (642,1204),
    "mode": "mixed",
    "n_events": 1000,
    "delta_t": 1000,
    "num_output_time_bins": 1,
    "sync_time": False,
    "flatten": True
}

lif_ros_config = {
    "vth": 30,
    "du": 2250,
    "dv": 1350
}

lif_default_config = {
    "vth": 35,
    "du": 1050,
    "dv": 1200
}

lif_fusion_config = {
    "vth": 2,
    "du": 1650,
    "dv": 1350
}

weights_default_config = {
    "w_ff": 255,
    "w_rec": 5,
    "w_o": 16,
    "kernel_size": 20
}

rosbag_default_config = {
    "depth_shape": (642,1204),
    "filename": "/home/altair/Postdoc/Codes/samples/rect_fast_cut.bag",
    "topic": "/depth_perception/depth",
    "sync": True,
    "resize": (64,120)
}

class Architecture:
   def __init__(self,num_steps: int,use_loihi2: bool) -> None:
      self.time_steps = num_steps
      self.use_loihi2 = use_loihi2
      self.prophesee_input_config = prophesee_input_default_config
      self.lif_config = lif_default_config
      self.lif_ros_config = lif_ros_config
      self.lif_fusion_config = lif_fusion_config
      self.bag_input_config = rosbag_default_config
      self.weights_config = weights_default_config

      if self.use_loihi2:
         self.run_cfg = Loihi2HwCfg(exception_proc_model_map=
                                 {PropheseeCameraSync: PyPropheseeCameraRawReaderModelSync})
      else:
         self.run_cfg_dvs = Loihi2SimCfg(
               exception_proc_model_map={Dense: PyDenseModelBitAcc,LIF: PyLifModelBitAcc,PropheseeCameraSync: PyPropheseeCameraRawReaderModelSync,BagReaderSync: LoihiPyBagReaderDVS})
         self.run_cfg_ros = Loihi2SimCfg(exception_proc_model_map={Dense: PyDenseModelBitAcc,LIF: PyLifModelBitAcc,BagReaderSync: LoihiPyBagReaderDVS},select_tag="fixed_pt")
         
      self._create_processes()
      self._connect_processes()
      
   def _create_processes(self) -> None:
      self.camera_dvs = PropheseeCameraSync(**self.prophesee_input_config)
      self.camera_ros = BagReaderSync(**self.bag_input_config)
      self.scaled_shape = (self.camera_dvs.shape[2],self.camera_dvs.shape[3])
      self.lif_dvs = LIF(shape=self.scaled_shape, **self.lif_config)
      self.lif_ros = LIF(shape=self.scaled_shape, **self.lif_ros_config)
      self.lif_fusion = LIF(shape=self.scaled_shape, **self.lif_fusion_config)
      self.py_receiver = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)
      self.py_receiver_ros = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)
      self.py_receiver_fusion = RingBuffer(shape=self.scaled_shape, buffer=self.time_steps)

   def _connect_processes(self) -> None:
      self.camera_dvs.s_out.reshape(self.scaled_shape).connect(self.lif_dvs.a_in)
      self.camera_dvs.t_out.connect(self.camera_ros.t_in)
      self.camera_ros.depth_out_port.connect(self.lif_ros.a_in)
      self.lif_dvs.s_out.connect(self.lif_fusion.a_in)
      self.lif_ros.s_out.connect(self.lif_fusion.a_in)
      #connect(self.camera_ros.depth_out_port, self.lif_ros.a_in,ops=[Weights(10)])
      self.lif_dvs.s_out.connect(self.py_receiver.a_in)
      self.lif_ros.s_out.connect(self.py_receiver_ros.a_in)
      self.lif_fusion.s_out.connect(self.py_receiver_fusion.a_in)

   def run(self):
      # Run
      run_cnd = RunSteps(num_steps=self.time_steps)
      run_cnd_ros = RunSteps(num_steps=30)
      self.camera_dvs.run(condition=run_cnd, run_cfg=self.run_cfg_dvs)

   def plot(self):
      # Get probed data from monitors
      print("gathering datas...")
      self.data_dvs = self.py_receiver.data.get().transpose()
      self.data_dvs = np.transpose(self.data_dvs,(0,2,1))
      self.data_ros = self.py_receiver_ros.data.get().transpose()
      self.data_ros = np.transpose(self.data_ros,(0,2,1))
      self.data_fusion = self.py_receiver_fusion.data.get().transpose()
      self.data_fusion = np.transpose(self.data_fusion,(0,2,1))
      print("shape dvs : ",self.data_dvs.shape)
      print("shape ros : ",self.data_ros.shape)
      print("shape fusion : ",self.data_fusion.shape)
      
      # Stop the execution of the network
      self.camera_dvs.stop()
      # Generate an animated plot from the probed data
      self.grid_shape = (self.scaled_shape[0],self.scaled_shape[1])
      print(self.grid_shape)
      self.grid_dvs = np.zeros(self.grid_shape)
      self.grid_ros = np.zeros(self.grid_shape)
      self.grid_fusion = np.zeros(self.grid_shape)
      #self.fig, self.axs = plt.subplots(2, 2)
      self.fig = plt.figure()
      gs = gridspec.GridSpec(2,2)
      self.ax0 = self.fig.add_subplot(gs[0,0])
      self.ax1 = self.fig.add_subplot(gs[0,1])
      self.ax2 = self.fig.add_subplot(gs[1,:])
      self.ax0.set_title("Dynamic Vision Sensor")
      self.ax1.set_title("Kinect Depth Camera")
      self.ax2.set_title("Fusion")
      self.im0 = self.ax0.imshow(self.grid_dvs, vmin=0.0, vmax=1.0,animated=True)
      self.im1 = self.ax1.imshow(self.grid_ros, vmin=0.0, vmax=1.0,animated=True)
      self.im2 = self.ax2.imshow(self.grid_fusion, vmin=0.0, vmax=1.0,animated=True)
      anim = matplotlib.animation.FuncAnimation(self.fig,self.update_lif, interval=40, init_func=self.setup_plot_lif,frames=self.time_steps, blit=True)
      plt.show()
      #anim.save('/home/altair/Postdoc/Videos/spiking_depth_dvs.mp4', writer = 'ffmpeg', fps = 30)

   def setup_plot_lif(self):
      self.im0.set_data(self.grid_dvs)
      self.im1.set_data(self.grid_ros)
      self.im2.set_data(self.grid_fusion)
      self.ax0.axis([0, self.grid_shape[1], 0.0, self.grid_shape[0]])
      self.ax1.axis([0, self.grid_shape[1], 0.0, self.grid_shape[0]])
      self.ax2.axis([0, self.grid_shape[1], 0.0, self.grid_shape[0]])
      # For FuncAnimation's sake, we need to return the artist we'll be using
      # Note that it expects a sequence of artists, thus the trailing comma.
      return self.im0, self.im1, self.im2, 

   def update_lif(self, i):
      self.grid_dvs = self.data_dvs[i]
      self.grid_ros = self.data_ros[i]
      self.grid_fusion = self.data_fusion[i]
      #self.grid = np.zeros(self.scaled_shape)
      self.im0.set_data(self.grid_dvs)
      self.im1.set_data(self.grid_ros)
      self.im2.set_data(self.grid_fusion)
      print(i)

      return self.im0, self.im1, self.im2,

if __name__ == '__main__':
   architecture = Architecture(120,False)
   #plt.ion()
   #plt.show(block=False)
   architecture.run()
   print("end")
   architecture.plot()
   #plt.show()
   
   
   
   