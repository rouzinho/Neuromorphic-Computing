# Sensor Fusion

In thes tutorial, we will demonstrate the use of the ROS-Loihi and EVK4-Loihi interfaces to perform sensor fusion. The Goal is to be able to track an object when it's in motion as well as to locate it while it remains static. The depth images are not temporally efficient due to the low frame rate but they remain precise to represent a static object. On the opposite, the event camera is endowed with a high temporal fidelity and computing efficiency for object in motion but is not adapted to keep a track on static objects.

## Code

For this example, we will use the ros-loihi and evk4-loihi versions that take into account the synchronization. Indeed, instead of running two parallel processes (one ros, one revk4), it is more efficient to begin by processing the events and send timestamps information. In turn, the ros-loihi interface will send depth images synchronized with the processed events. This requires for both events and depth recordings to begin at the same time.

```
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
```
We begin by defining the interfaces parameters, they remain the same as in the standalone versions.

```
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
```
Here, the connections are the followings : DVS -> ROS-Loihi->LIF_ROS, DVS->LIF_DVS, LIF_DVS and LIF_ROS -> LIF_FUSION.

The result :

![](https://github.com/rouzinho/Neuromorphic-Computing/blob/main/img/spiking_fusion.gif?raw=true)

The complete code is available inside the [sensor_fusion folder](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/sensor_fusion). Don't forget to copy the bags_reader and dvs folders inside to be able to use the interfaces.