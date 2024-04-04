# control_sandbox

Sandbox code to play with (Cartesian) controllers using the [MuJoCo simulator](https://github.com/google-deepmind/mujoco/) and [Spatial Maths for Python](https://github.com/bdaiinstitute/spatialmath-python). The controllers are formulated using exponential coordinates, a good reference is the textbook [1] which is online available [here](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf).

## Installation

It is recommended to set up a new virtual environment in Python (tested with venv) and install the requirements in the project's root folder via
 ```bash
pip install -r requirements.txt
```

## Usage

Simply run the scripts in the root folder in a shell or debugger, e.g.:
 ```bash
python cartesian_impedance_control.py
```

## Controllers

The controller implementations are located in the `$ROOT/controllers` directory.

### Simple Cartesian Impedance Controller

This implements a simple Cartesian Impedance Controller using the implementation in [1], p. 444, Eq. (11.65) without full arm dynamcis compensation (only gravitational load and coriolis / centripetal forces are compensated) and a virtual mass of M=0. Note, that this can be seen as a feedforward force controller (cf. [1], p. 435, Eq. (11.50)), where the desired external end-effector wrench is specified by a cartesian impedance tracking law that closes a feedback loop on measured joint positions and velocities.

### FAQ
#### Installation and running on OSX
To run and debug the project with VScode on OSX, you need to run it with the mjpython interperter. Use the "⌘ + Shift + P" to open configuration entry for the "Python: Select Interpreter" and fill in the "Enter interpret path: " with the mjpython location, for example "./.venv/bin/mjpython".

## References

[1] ... Lynch, K. M., & Park, F. C. (2017). Modern robotics. Cambridge University Press.

