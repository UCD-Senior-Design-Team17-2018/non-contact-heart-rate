from picamera import PiCamera
from time import sleep
camera = PiCamera()
camera.framerate = 30
camera.exposure_mode = 'fixedfps' 
camera.start_preview()
camera.start_recording('/home/pi/video.h264')
sleep()
camera.stop_recording()
camera.stop_preview()
