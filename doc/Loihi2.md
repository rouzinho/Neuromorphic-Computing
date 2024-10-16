# Intel Neuromorphic Chip

This page will introduce neuromorphic computing with Intel Loihi 2.

## Loihi 2

Loihi 2 is Intel's latest neuromorphic research chip, implementing spiking neural networks with programmable dynamics, modular connectivity, and optimizations for scale, speed, and efficiency. The open source framework [LAVA](https://lava-nc.org/) support the simulation and computing of a wide range of neural networks directly on the Loihi 2 or on a CPU. The programming langage used is Python 3.

## Installation

In order to run code on neuromorphic chip, we will need several libraries developed by Intel. Unfortunately, the documentation only details the installation of these libraries separately but we propose here to gather them under the same virtual environment. We will install the main framework that is in charge of simulating spiking neural networks. In addition, we will install lava-dnf which run dynamic neural fields (DNF) as spiking neural networks with reccurent connections. The theory behind DNFs can be found on the [dynamic field theory](https://dynamicfieldtheory.org/) website. Finally, we will install the lava-peripherals library who provide support to interface Dynamic Vision Sensor directly with Loihi 2.

First create a virtual environment :
```
python3 -m venv neuro && . neuro/bin/activate
python3 -m pip install --upgrade pip
```
Install the main framework :
```
git clone https://github.com/lava-nc/lava.git
cd lava
poetry export --without-hashes -f requirements.txt -o requirements.txt
pip install --no-cache-dir --no-deps -r requirements.txt
pip install .
cd ..
```
Install the DNF library :
```
git clone https://github.com/lava-nc/lava-dnf.git
cd lava-dnf
poetry export --without-hashes -f requirements.txt -o requirements.txt
pip install --no-cache-dir --no-deps -r requirements.txt
pip install .
```
Install the lava peripherals :
```
cd ..
git clone https://github.com/lava-nc/lava-peripherals.git
cd lava-peripherals
poetry export --without-hashes -f requirements.txt -o requirements.txt
pip install --no-cache-dir --no-deps -r requirements.txt
pip install .
```
Allow your virtual environment to access system libraries. It will support the use of the metavision SDK inside your code.
```
cd ..
cd neuro/
nano pyvenv.cfg
#edit this line to allow access of the python system library
include-system-site-packages = true
#resource your environment
source ~/neuro/bin/activate
```



