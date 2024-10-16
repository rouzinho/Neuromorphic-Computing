#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from lava.proc.lif.process import LIF # type: ignore
from lava.proc.monitor.process import Monitor
from lava.proc.monitor.models import PyMonitorModel
from lava.proc.dense.process import Dense
from lava.proc.io.sink import RingBuffer
from lava.proc.io.extractor import VarWire
from lava.proc.io.utils import ChannelConfig, SendFull
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.lib.dnf.connect.connect import connect
from lava.lib.dnf.operations.operations import Weights, Convolution
from lava.lib.dnf.inputs.gauss_pattern.process import GaussPattern
from lava.lib.dnf.inputs.rate_code_spike_gen.process import RateCodeSpikeGen
from lava.lib.dnf.kernels.kernels import Kernel
from lava.lib.dnf.kernels.kernels import MultiPeakKernel
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.proc.dense.process import Dense
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.decorator import implements
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.model.py.ports import PyRefPort
from lava.lib.dnf.operations.operations import (
    Weights,
    ExpandDims,
    ReorderDims,
    ReduceAlongDiagonal,
    ExpandAlongDiagonal,
    Flip)
from lava.utils.system import Loihi2
Loihi2.preferred_partition = 'oheogulch'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
    
else:
    print("Loihi2 compiler is not available in this system. "
          "This tutorial will execute on CPU backend.")


class ArchitectureDNF:

    def __init__(self,
                 kernel: Kernel,time_steps,size_gauss) -> None:
        shape = (size_gauss,)
        self.int_shape = size_gauss
        self.time_steps = time_steps
        # Instantiate Processes
        self.gauss_1 = GaussPattern(shape=shape,
                                            amplitude=0,
                                            mean=10.25,
                                            stddev=2.0)
        self.gauss_2 = GaussPattern(shape=shape,
                                            amplitude=0,
                                            mean=39.0,
                                            stddev=2.0)

        self.spike_generator_1 = RateCodeSpikeGen(shape=shape)
        self.spike_generator_2 = RateCodeSpikeGen(shape=shape)
        self.input_dense1 = Dense(weights=np.eye(shape[0]) * 25)
        self.input_dense2 = Dense(weights=np.eye(shape[0]) * 25)
        self.input_dense3 = Dense(weights=np.eye(shape[0]) * 25)
        self.dnf1 = LIF(shape=shape, du=409, dv=2045, vth=200)
        self.dnf2 = LIF(shape=shape, du=409, dv=2045, vth=200)
        self.projection = LIF(shape=shape + shape, du=409, dv=2045, vth=200)
        self.py_receiver_dnf1 = RingBuffer(shape=shape, buffer=self.time_steps)
        self.py_receiver_dnf2 = RingBuffer(shape=shape, buffer=self.time_steps)
        self.py_receiver_dnftf = RingBuffer(shape=shape+shape, buffer=self.time_steps)
        self.run_continuous = RunContinuous()
        
        # When running on Loihi 2 we additionally have spike injector and reader CProcesses. 
        # They bridge Python and NC-Processes
        if loihi2_is_available:
            self.injector1 = PyToNxAdapter(shape=shape) # type: ignore
            self.injector2 = PyToNxAdapter(shape=shape) # type: ignore
            self.spike_reader = NxToPyAdapter(shape=shape) # type: ignore
        
        # Make Connections of the Network
        self.gauss_1.a_out.connect(self.spike_generator_1.a_in)
        self.gauss_2.a_out.connect(self.spike_generator_2.a_in)

        if loihi2_is_available:
            self.spike_generator_1.s_out.connect(self.injector1.inp)
            self.spike_generator_2.s_out.connect(self.injector2.inp)
            self.injector1.out.connect(self.input_dense1.s_in)
            self.injector2.out.connect(self.input_dense2.s_in)
        else:
            self.spike_generator_1.s_out.connect(self.input_dense1.s_in)
            self.spike_generator_2.s_out.connect(self.input_dense2.s_in)

        self.input_dense1.a_out.connect(self.dnf1.a_in)
        self.input_dense2.a_out.connect(self.dnf2.a_in)
        connect(self.dnf1.s_out, self.dnf1.a_in, [Convolution(kernel)])
        connect(self.dnf2.s_out, self.dnf2.a_in, [Convolution(kernel)])
        connect(self.dnf1.s_out, self.projection.a_in,ops=[ExpandDims(new_dims_shape=shape[0]),Weights(10)])
        connect(self.dnf2.s_out, self.projection.a_in,ops=[ExpandDims(new_dims_shape=shape[0]),ReorderDims(order=(1, 0)),Weights(10)])
        kernel_sus = MultiPeakKernel(amp_exc=58,
                         width_exc=[3,3],
                         amp_inh=-30,
                         width_inh=[6,6])
        connect(self.projection.s_out, self.projection.a_in, [Convolution(kernel_sus)])
        

        if loihi2_is_available:
            self.dnf1.s_out.connect(self.spike_reader.inp)
            #self.spike_reader.out.connect(self.py_receiver.a_in)
        else:
            self.dnf1.s_out.connect(self.py_receiver_dnf1.a_in)
            self.dnf2.s_out.connect(self.py_receiver_dnf2.a_in)
            self.projection.s_out.connect(self.py_receiver_dnftf.a_in)

        # Set up a run configuration
        if loihi2_is_available:
            self.run_cfg = Loihi2HwCfg()
        else:
            self.run_cfg = Loihi1SimCfg(select_tag="fixed_pt")

    def run(self):
        condition1 = RunSteps(num_steps=self.time_steps)
        condition2 = RunSteps(num_steps=50)
        condition3 = RunSteps(num_steps=100)
        
        #self.gauss_1.amplitude = 10000
        self.gauss_1.run(condition=condition3, run_cfg=self.run_cfg)
        self.gauss_2.run(condition=condition3, run_cfg=self.run_cfg)
        #self.gauss_position.run(condition=self.run_continuous, run_cfg=self.run_cfg)
        self.gauss_1.amplitude = 7000
        self.gauss_2.amplitude = 7000
        #self.gauss_1.amplitude = 3000
        #self.gauss_1.mean = 20.0
        self.gauss_1.run(condition=condition2, run_cfg=self.run_cfg)
        self.gauss_2.run(condition=condition2, run_cfg=self.run_cfg)
        self.gauss_1.amplitude = 0
        self.gauss_2.amplitude = 0
        #self.gauss_1.amplitude = 3000
        #self.gauss_1.mean = 20.0
        self.gauss_1.run(condition=condition3, run_cfg=self.run_cfg)
        self.gauss_2.run(condition=condition3, run_cfg=self.run_cfg)

    def plot(self):
        self.data_dnf1 = self.py_receiver_dnf1.data.get().transpose()
        self.data_dnf2 = self.py_receiver_dnf2.data.get().transpose()
        self.data_dnftf = self.py_receiver_dnftf.data.get().transpose()
        #print(self.data_dnf1[0])
        #print(self.data_dnftf[0])
        self.gauss_1.stop()
        self.gauss_2.stop()

        self.x_gauss1 = np.arange(self.int_shape)
        self.y_gauss1 = np.zeros(self.int_shape)
        self.x_gauss2 = np.arange(self.int_shape)
        self.y_gauss2 = np.zeros(self.int_shape)
        self.shape_grid = (self.int_shape,self.int_shape)
        self.grid = np.zeros(self.shape_grid)
        self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(3,1,gridspec_kw={'height_ratios': [20, 3,3]})
        self.fig.tight_layout()
        self.ax0.set_title("Projection of Gauss 1 and 2")
        self.ax1.set_title("Gauss 1")
        self.ax2.set_title("Gauss 2")
        self.im = self.ax0.imshow(self.grid, vmin=0.0, vmax=1.0,animated=True)
        self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update_gauss, interval=10, init_func=self.setup_plot_gauss, frames=self.time_steps, blit=True)
        plt.show()
        #self.ani.save('/home/altair/Postdoc/Codes/sim_dnf/wm.mp4', writer = 'ffmpeg', fps = 30)

    def setup_plot_gauss(self):
        self.im.set_data(self.grid)
        self.g1, = self.ax1.plot(self.x_gauss1,self.y_gauss1, color='red')
        self.g2, = self.ax2.plot(self.x_gauss2,self.y_gauss2, color='blue')
        self.ax0.axis([0, self.int_shape-1, 0.0, self.int_shape-1])
        self.ax1.axis([0, self.int_shape-1, 0.0, 1.5])
        self.ax2.axis([0, self.int_shape-1, 0.0, 1.5])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.im, self.g1, self.g2,

    def update_gauss(self, i):
        print(i)
        self.y_gauss1 = self.data_dnf1[i]
        self.y_gauss2 = self.data_dnf2[i]
        self.grid = self.data_dnftf[i]
        self.im.set_data(self.grid)
        self.g1.set_ydata(self.y_gauss1)
        self.g2.set_ydata(self.y_gauss2)

        return self.im, self.g1, self.g2,

        
if __name__ == '__main__':
    detection_kernel = MultiPeakKernel(amp_exc=83, width_exc=3.75, amp_inh=-70, width_inh=7.5)
    architecture = ArchitectureDNF(detection_kernel,250,50)
    
    architecture.run()
    architecture.plot()
        