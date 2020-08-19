# JAPARI : Job and progress alternate inference middleware

<!-- ABOUT THE PROJECT -->
## Overview

JAPARI is a middleware stack for doing inference on energy-harvesting intermittent systems. We implemented our JAPARI design on the Texas Instruments MSP430FR5994 LaunchPad, and used the internal low-energy accelerator (LEA) hardware for DNN inference acceleration. JAPARI consists of multiple inference functions, which are exposed via an intuitive API. Each inference function contains four major design components: 
* the data mover.
* the kernel generator. 
* the channel generator.
* the progress seeker.

The data mover fetches/preserves the IFM/OFM and footprints. The kernel and channel generators append footprints kernels and channels onto the IFM tile and weight kernel tiles. Upon power resumption, the progress seeker searches for the latest footprint across all preserved footprints in the preservation buffer in NVM.

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)
* [Using JAPARI](#using-japari)
  
<!--* [Contributing](#contributing)-->

## Directory/File Structure
Below is an explanation of the directories/files found in this repo. 
```
├── driverlib
│   └── ...
├── dsplib
│   └── ...
├── libJAPARI
│   ├── src
│   │   ├── japari.c
│   │   ├── convolution.c
│   │   ├── fc.c
│   │   └── nonlinear.c
│   ├── japari.h
│   ├── convolution.h
│   ├── fc.h
│   └── nonlinear.h
├── main.c
├── main.h
└── model.h
```
`driverlib/` is a set of drivers produced by Texas Instruments for accessing the peripherals found on the MSP430 family of microcontrollers. 

`dsplib/` is a set of highly optimized functions produced by Texas Instruments to perform many common signal processing operations on fixed-point numbers for MSP430 microcontrollers. 

`libJAPARI/` contains source code for basic neural network operations for JAPARI. 

`main.c` contains an example to run a DNN inference.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Here is the basic software and hardware you need to build/run the provided example. 

* [Code composer studio](http://www.ti.com/tool/CCSTUDIO "link") (recommended versions: > 7.0)
* [MSP Driver Library](http://www.ti.com/tool/MSPDRIVERLIB "link")
* [MSP DSP Library](http://www.ti.com/tool/MSP-DSPLIB "link")
* [MSP-EXP430FR5994 LaunchPad](http://www.ti.com/tool/MSP-EXP430FR5994 "link")

### Setup and Build

1. Download/clone this repository
2. Download `Driver` & `DSP` library from http://www.ti.com/ 
3. Import this project to your workspace of code composer studio (CCS). 
4. Add `PATH_TO_DSPLIB` & `PATH_TO_DIRVERLIB` to library search path


