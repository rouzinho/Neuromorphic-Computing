# EVK4 interface

This page introduces the interface to read data events from the Prophesee EVK4 and send them to Loihi 2. There are two classes that can create an interface. The first one is provided by the lava-peripherals [package](https://github.com/lava-nc/lava-peripherals) and the second is an extension of the lava-peripherals that can support a synchronization with a depth sensor.
The package can be found [here](https://github.com/rouzinho/Neuromorphic-Computing/tree/main/src/dvs)

## EVK4 in standalone

The process is quite simple and we advise following the tutorial [here](https://github.com/lava-nc/lava-peripherals/blob/main/tutorials/lava/lib/peripherals/dvs/PropheseeCamera.ipynb).

## EVK4 with depth synchronization

The class is a minor extension of lava-peripherals. It consists of sending timestamps to any connected modules so it becomes possible to synchronize events with depth frames. This avoid starting parallel processes for the depth and the event camera. Here, the DVS first process events and triggers the process of depth frames at the same time. This renders the execution on Loihi less complicated.

We will demonstrate how to use these classes in the tutorial section.


