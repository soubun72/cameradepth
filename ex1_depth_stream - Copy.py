
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
from primesense import openni2
from primesense import _openni2 as c_api
import time

step = 5;

x_data = np.zeros((1,76800))
y_data = np.zeros((1,76800))
z_data = np.zeros((1,76800))
i = 0

for x in range(0,239,step - 1):
        for y in range(0,319,step - 1):
            x_data[0,i] = x
            y_data[0,i] = y
            i = i + 1

i = 0

openni2.initialize()     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()

rgb_stream = dev.create_color_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))
rgb_stream.start()

depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
depth_stream.start()

def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb  
    
def get_depth():
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255    
    Note1: 
        fromstring is faster than asarray or frombuffer
    Note2:     
        .reshape(120,160) #smaller image for faster response 
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    # Shown unknowns in black
    d4d = 255 - d4d    
    return dmap, d4d
#get_depth


done = False

plt.ion()

while not done:
    key = cv2.waitKey(1)
    ## Read keystrokes
    key = cv2.waitKey(1) & 255
    if key == 27: # terminate
      print ("\tESC key detected!")
      done = True
    
    rgb = get_rgb()
    cv2.imshow('rgb', rgb)     
    dmap,d4d = get_depth()
    #print 'Center pixel is {}mm away'.format(dmap[119,159])

    ## Display the stream syde-by-side
    cv2.imshow('depth', d4d)    

    ax = plt.axes(projection = "3d")
    #ax.plot()

    for x in range(0,239,step-1):
        for y in range(0,319,step-1):
            z_data[0,i] = 1000 - dmap[x,y]
            i = i + 1
            #print(dmap[x,y])

    i = 0

    ax.scatter(x_data, y_data, z_data)
    ax.set_xlim([0, 320])
    ax.set_ylim([0, 240])
    ax.set_zlim([0, 1000])
    plt.draw()
    plt.pause(0.05)
    ax.cla()


cv2.destroyAllWindows()
depth_stream.stop()
rgb_stream.stop()
openni2.unload()
