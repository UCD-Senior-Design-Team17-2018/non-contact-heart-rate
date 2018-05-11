import cv2
import datetime
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Change these variables based on the location of your cloned, local repositories on your computer
#PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
#face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used

#yay no more changing path everytime for haarclassfier
#get path for haarclassifier file
#haarclassifer must be present because no check
HAARCLASSIFIER = 'haarcascade_frontalface_default.xml'
haarpath = os.getcwd() + '/' + HAARCLASSIFIER # get path of harr xml file
face_cascade = cv2.CascadeClassifier(haarpath) # Full path must be used

firstFrame = None
time = []
R = []
G = []
B = []
pca = FastICA(n_components=3) #the ICA class
#cap = cv2.VideoCapture(0) # open webcam

cap = cv2.VideoCapture("C:\\Users\\Bijta\\Documents\\GitHub\\non-contact-heart-rate\\video_analysis\\test\\DSC_0009.mov")
#if cap.isOpened() == False:
#    print("Failed to open webcam")

frame_num = 0 # start counting the frames
plt.ion() # interactive

# Read until video is completed
while cap.isOpened(): 
	# capture frame by frame
    ret, frame = cap.read() # read in the frame, grabs next frame and frame status/if frame exists
    if ret == True: # if status is true
        frame_num += 1 # counts frame number
        if frame_num == 1:
            rectROI = cv2.selectROI(frame) #self-select rectangular ROI 
            #top left is (0,0) usually
            print('ROI coordinates ')
            print(rectROI) 
			
        if firstFrame is None: 
            start = datetime.datetime.now() # start time
            time.append(0)
            firstFrame = frame
            cv2.imshow("frame",firstFrame)

            #old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY) #converts RGB to grayscale for VJ
            #faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) # Use Viola-Jones classifier to detect face
            if rectROI == ():
                firstFrame = None
            else:
                
                #for (x,y,w,h) in rectROI: # selectROI outputs x,y, width and height
                #x2 = x+w # other side of rectangle, x
                #y2 = y+h # other side of rectangle, y
                x = rectROI[0]
                y = rectROI[1]
                w = rectROI[2]
                h = rectROI[3]
                cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2) #draw rect.
                cv2.imshow("frame",firstFrame)
				
                # Make a mask
                VJ_mask = np.zeros_like(firstFrame) 
                VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                
                ROI = VJ_mask
                ROI_color = cv2.bitwise_and(ROI,ROI,mask=ROI) 
                cv2.imshow('ROI',ROI_color)
                
                #take average signal in the region of interest
                R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI) 
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        else:
            current = datetime.datetime.now()-start
            # time for the current frame
            current = current.total_seconds()
            time.append(current)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
            cv2.imshow('ROI',ROI_color)
            #take average signal in the region of interest (mask)
            R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
            R.append(R_new)
            G.append(G_new)
            B.append(B_new)
            if frame_num >= 900: # when 900 frames collected, start calculating heart rate (sliding window)
                N = 900
                    #normalize RGB signals
                G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1)) 
                G_std = G_std.reshape(1, -1)[0]
                R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1)) 
                R_std = R_std.reshape(1, -1)[0]
                B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                B_std = B_std.reshape(1, -1)[0]
                #   G_std = np.array(G[-(N-1):])
                #R_std = np.array(R[-(N-1):])
                #B_std = np.array(B[-(N-1):])
               # T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
                T = 1/29.97 
               # do ICA (called PCA because originally tried PCA)
                X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
                #X_f = (R_std+G_std+B_std)/3
                b, a = signal.butter(4, [0.5/15, 3/15], btype='band') #Butterworth filter
                #X_f = signal.lfilter(b, a, X_f[0]) 
                N = len(np.pad(X_f[1],(0,1024),'constant'))
                yf = fft(np.pad(X_f[1],(0,1024),'constant'))
                yf = yf/np.sqrt(N) #Normalize FFT
                xf = fftfreq(N, T) # FFT frequencies 
                xf = fftshift(xf) #FFT shift
                yplot = fftshift(abs(yf))
                plt.figure(1)
                plt.gcf().clear()
                fft_plot = yplot
                # Find highest peak between 0.75 and 4 Hz 
                fft_plot[xf<=0.75] = 0 
                print(str(xf[(xf>=0) & (xf<=4)][fft_plot[(xf>=0) & (xf<=4)].argmax()]*60)+' bpm') # Print heart rate
                plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT
                plt.pause(0.001)
            if frame_num % 10 == 0:
                print(frame_num)
            if frame_num==1100:
                cap.release()
                raise KeyboardInterrupt