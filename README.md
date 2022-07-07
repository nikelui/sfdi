# sfdi
A collection of Python scripts to acquire SFDI data and process it, in order to measure optical properties of tissue.

**NOTE:** this project was developed for personal use, it is not optimized to be distributed
and used in other machines. An extensive (and slow) re-work is currently in progress to make
the library easier to install (e.g. frozen binaries) and to write better documentation.
Any contribution is welcome.

## Dependencies

- Numpy
- Scipy
- Matplotlib
- tkinter
- OpenCv
- [cvui](https://dovyski.github.io/cvui/#:~:text=cvui%20is%20a%20(very)%20simple,it%20OpenGL%20enabled%2C%20for%20instance.) (already included)
- PyInstaller (to build frozen binaries)
...

## Acquisition
**Coming soon**

## Processing
**Coming soon**

## Cameras
The SFDI module provides a camera class object that acts as a wrapper for the
actual camera API code. For each camera you plan to use in the system, you need to
install the respective drivers/SDK and implement the camera class methods
(make a copy of DummyCam.py and read the descriptions / return values).
### Currently implemented cameras
- **DummyCam**: blank camera class with placeholders
- **Generic**: OpenCV generic VideoCapture I/O module
- **pointGrey**: old FLIR FlyCapture 
[SDK](https://www.flir.com/products/flycapture-sdk/?vertical=machine%20vision&segment=iis)
  - NOTE: this is obsolete software (last version is from 2019?). FLIR has released a new
	SDK called [Spinnaker](https://www.flir.com/products/spinnaker-sdk/?vertical=machine%20vision&segment=iis),
  which supports their new USB3 / GiGE cameras.
- **piCam**: RaspbberryPi piCamera module (only works on linux arm system for RaspberryPi)
- **IS**: Imaging Source camera [SDK](https://github.com/TheImagingSource)
- **xiCam** [coming soon]: ximea camera [SDK](https://www.ximea.com/support/wiki/apis/Python)

## Planned
- Substitute OpenCV / cvui with native python libraries
  - Use PIL for image processing / ROI selection
  - Use TkInter for GUI elements
- OpenCL for GPU parallel processing