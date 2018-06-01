try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing RPi.GPIO!  This is probably because you need superuser privileges.  You can achieve this by using 'sudo' to run your script")
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.IN)
try:
    while True:
        if GPIO.input(25):
	    print("LOW BATTERY! Going to shutdown!")
	else:
	    print('Battery is fine!')
	sleep(1)
finally:
    GPIO.cleanup()