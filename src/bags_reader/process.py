import sys
import numpy as np
import time
from threading import Thread
import cv2
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage

class BagReader(AbstractProcess):
   """
    Process that read depth images from ros bag file and send them through outport 

    Parameters
    ----------
    filename: str
        String to filename
    topic: str
        Topic of the bag from where to retrieve the datas.
    sensor_shape: tuple
        Shape of the image / kinect camera.
    """
   def __init__(
            self,
            depth_shape: tuple,
            filename: str = "",
            topic: str = "",
            resize: tuple = None) -> None:
      if filename == "":
         raise ValueError(
            "you must provide the path of a bag"
         )
      if topic == "":
         raise ValueError(
            "you must provide a topic to listen to"
         )
      self.filename = filename
      self.topic = topic
      self.resize = resize
      if resize is not None:
         self.depth_shape = resize
      else:
         self.depth_shape = depth_shape
      self.depth_out_port = OutPort(shape=self.depth_shape)
      super().__init__(
         depth_shape=self.depth_shape,
         filename=self.filename,
         topic=self.topic,
         resize=self.resize
      )

#class implementing the bag reader
@implements(proc=BagReader, protocol=LoihiProtocol)
@requires(CPU)
class LoihiDensePyBagReader(PyLoihiProcessModel):
   depth_out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

   def __init__(self, proc_params: dict) -> None:
      super().__init__(proc_params)

      self._file_name = proc_params["filename"]
      self._path = Path(self._file_name)
      self._typestore = get_typestore(Stores.ROS1_NOETIC) 
      self._topic = proc_params["topic"]
      self._depth_shape = proc_params["depth_shape"]
      self._resize = proc_params["resize"]
      self._ind_f, self._ind_l = self._init_bag()
      self._current_ind = self._ind_f

   def run_spk(self):
      #print("get image")
      depth_image = self._get_bag_frame()
      self.depth_out_port.send(depth_image)
      self._current_ind += 1

   def _get_bag_frame(self):
      img = None
      if self._current_ind >= self._ind_l:
         self._current_ind = self._ind_l
      with AnyReader([self._path], default_typestore=self._typestore) as reader:
         connections = [x for x in reader.connections if x.topic == self._topic]
         for connection, timestamp, rawdata in reader.messages(connections=connections):
               msg = reader.deserialize(rawdata, connection.msgtype)
               if msg.header.seq == self._current_ind:
                  img = message_to_cvimage(msg)
                  if self._resize is not None:
                     img = cv2.resize(img, (self._depth_shape[1],self._depth_shape[0]),  interpolation = cv2.INTER_NEAREST)
               
      return img

   #return first and last index of msg (seq doesn't start at 0)
   def _init_bag(self):
      first = True
      ind_f = 0
      count = 0
      with AnyReader([self._path], default_typestore=self._typestore) as reader:
         count = reader.message_count
         connections = [x for x in reader.connections if x.topic == self._topic]
         for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            #print(msg)
            if first:
               ind_f = msg.header.seq
               break
      ind_l = ind_f + count-1

      return ind_f,ind_l
   
   def _pause(self):
      """Pause was called by the runtime"""
      super()._pause()
      self.t_pause = time.time_ns()

   def _stop(self) -> None:
      """Stop was called by the runtime"""
      super()._stop()
      