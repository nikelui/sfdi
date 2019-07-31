# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:26:31 2019

@author: luibe59
"""
import sys
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import PyCapture2 as pc

def setCamera_pg(num=0,res='1280p',fps=15):
    """Set up and configure a Point Grey camera using FlyCapture proprietary drivers
(uses Point Grey PyCapture2 Python wrapper).

syntax: cam = setCamera_pg(num,res)

- num -> an integer containing the camera index. Used to identify camera with getCameraFromIndex()
- res -> a string specifying the resolution for acquisition. Accepted values are '1280p' (1280x960)
         and '640p' (640x480). Use the second one for a framerate higher than 15fps.

- cam -> a PyCapture2 Camera object, connected to the current camera.

NOTE: for now, most camera configurations are hard-coded in this module

"""
    ## Set the default algorithm for color processing
    pc.setDefaultColorProcessing(pc.COLOR_PROCESSING.HQ_LINEAR)
    
    bus = pc.BusManager() # constructor
    try:
        uid = bus.getCameraFromIndex(num) # this retrieves the address of the camera to connect
    except pc.Fc2error as fc2Err:
            print('Error retrieving device address: %s' % fc2Err)
            raise SystemExit(-1)
    cam = pc.Camera() # Camera object
    try:
        cam.connect(uid) # connect the camera
    except pc.Fc2error as fc2Err:
            print('Error connecting camera: %s' % fc2Err)
            raise SystemExit(-1)

    ## Here set the acquisition properties (look at the manual for supported values)
    if res == '1280p':
        vMode = pc.VIDEO_MODE.VM_1280x960Y8 # 1280x960 pixels, RAW 8bit
    elif res == '640p':
        vMode = pc.VIDEO_MODE.VM_640x480Y8 # 640x480 pixels, RAW 8bit
    else:
        vMode = pc.VIDEO_MODE.VM_640x480Y8 # default to 640x480 pixels, RAW 8bit

    if fps == 60:
        fRate = pc.FRAMERATE.FR_60 # 60fps. This limits the exposure to 16ms
    elif fps == 30:
        fRate = pc.FRAMERATE.FR_30 # 30fps. This limits the exposure to 33ms
    elif fps == 15:
        fRate = pc.FRAMERATE.FR_15 # 15fps. This limits the exposure to 66ms
    else:
        fRate = pc.FRAMERATE.FR_7_5 # Default to 7.5fps. This limits the exposure to 133ms
    
    try:
        cam.setVideoModeAndFrameRate(vMode,fRate)
    except pc.Fc2error as fc2Err:
         print('Error setting video mode and framerate: %s' % fc2Err)
         
    ## Here set camera properties (like disabling auto-exposure)
    ## Exposure time should be controlled in a GUI
    ## setProperty does not work properly if you don't pass a Property object
    prop = pc.Property(pc.PROPERTY_TYPE.BRIGHTNESS,present=True,absControl=False,valueA=0)
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.AUTO_EXPOSURE,absControl=True,autoManualMode=False,onOff=False)
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.GAMMA,absControl=True,absValue=1.0,onOff=True)
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=10.0,autoManualMode=False)
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.GAIN,absControl=True,absValue=0.0,autoManualMode=False)
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,absControl=True,absValue=fps,autoManualMode=False,onOff=True)
    if fps == 0:
        prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,onOff=False) # Disable fps, to get longer exposure time
    cam.setProperty(prop) # DONE
    prop = pc.Property(pc.PROPERTY_TYPE.WHITE_BALANCE,onOff=False)
    cam.setProperty(prop) # DONE

# This does not work properly
#    cam.setProperty(type=pc.PROPERTY_TYPE.BRIGHTNESS,absValue=0.0,onOff=False)
#    cam.setProperty(type=pc.PROPERTY_TYPE.GAIN,absValue=0.0,onOff=False)
#    
#    cam.setProperty(type=pc.PROPERTY_TYPE.GAIN,onOff=False) # Turn all features off, except for exp time
#    cam.setProperty(type=pc.PROPERTY_TYPE.AUTO_EXPOSURE,onOff=False)
#    cam.setProperty(type=pc.PROPERTY_TYPE.FRAME_RATE,autoManualMode=False)
#    cam.setProperty(type=pc.PROPERTY_TYPE.SHUTTER,autoManualMode=False)
#    cam.setProperty(type=pc.PROPERTY_TYPE.SHUTTER,absControl=True) # set real world unit (ms)
#    cam.setProperty(type=pc.PROPERTY_TYPE.WHITE_BALANCE,onOff=False) # turn white balance off

    cam.setConfiguration(
        numBuffers=10,
        grabMode=pc.GRAB_MODE.BUFFER_FRAMES)

    print('Camera connected. Remember to call cam.disconnect() in your script.')
    return cam