## EDIT (4/11/2018): Some interesting things noted:
So the code has been commented, as requested. I also tried the code with just a simple background, and still there is a 50bpm peak in the FFT, with visible oscillatory behavior in the RGB signal, even though it's supposed to be constant. So something is wrong with the camera maybe? If you want the code that works on the background, please let me know, as this current code will not do so because it will only proceed when it detects a face.

# Background
I am trying to develop a RGB camera-based non-contact measurement of heart rate using principles of remote photoplethysmography. I want to be able to measure a range of heart rates I have been looking into multiple methods, which are all quite similar though. Typically, they all detect/track the face, and then they spatially average the colors in the facial region. They either do a FFT of the green or red temporal signal, or they do the FFT of a component coming from a blind-source separation method (like ICA or PCA).

# Problem
Now, I have been working on this, however I have noticed that the frequency seems to always stay around 50bpm (my heart rate is usually around 70bpm), occasionally jumping to the correct bpm. I have even checked this with a fitness tracker. I have tried adding filters, detrending, etc. but with no success. What could be the problem here? Could there be some sort of other frequencies I am not considering here from the environment? I have tried this out in various environments, and illuminations and that does not seem to affect the problem. Is FFT not robust enough for this method? I have tried Welch's PSD, which seems like it gives better results, but still has some bias towards 50 bpm (0.83 Hz). 

Here is some code based on [this][1] paper:

    # -*- coding: utf-8 -*-
    """
    Created on Mon Apr  9 13:55:28 2018

    @author: Bijta
    """

    import cv2
    import datetime
    import numpy as np
    from scipy import signal
    from scipy.fftpack import fft, fftfreq, fftshift
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    # Change these variables based on the location of your cloned, local repositories on your computer
    PATH_TO_HAAR_CASCADES = "C:/Users/Bijta/Documents/GitHub/non-contact-heart-rate/video_analysis/" 
    face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used
    firstFrame = None
    time = []
    R = []
    G = []
    B = []
    pca = FastICA(n_components=3) #the ICA class
    cap = cv2.VideoCapture(0) # open webcam
    if cap.isOpened() == False:
        print("Failed to open webcam")
    frame_num = 0 # start counting the frames
    plt.ion() # interactive
    while cap.isOpened(): 
        ret, frame = cap.read() # read in the frame, status
        if ret == True: # if status is true
            frame_num += 1 # count frame
            if firstFrame is None: 
                start = datetime.datetime.now() # start time
                time.append(0)
                # Take first frame and find face in it
                firstFrame = frame
                cv2.imshow("frame",firstFrame)
                old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) # Use Viola-Jones classifier to detect face
                if faces == ():
                    firstFrame = None
                else:
                    for (x,y,w,h) in faces: # VJ outputs x,y, width and height
                        x2 = x+w # other side of rectangle, x
                        y2 = y+h # other side of rectangle, y
                        cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2) #draw rect.
                        cv2.imshow("frame",firstFrame)
                        # Make a mask
                        VJ_mask = np.zeros_like(firstFrame) 
                        VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                        VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                    ROI = VJ_mask
                    ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask) 
                    cv2.imshow('ROI',ROI_color)

                    #take average signal in the region of interest (mask)
                    R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI) 
                    R.append(R_new)
                    G.append(G_new)
                    B.append(B_new)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            else:
                # time for the current frame
                current = datetime.datetime.now()-start 
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
                    T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)])) #calculate time between first and last frame (period)
                    # do ICA (called PCA because originally tried PCA)
                    X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose() 
                    b, a = signal.butter(4, [0.5/15, 1.6/15], btype='band') #Butterworth filter
                    X_f = signal.lfilter(b, a, X_f) 
                    N = len(X_f[0])
                    yf = fft(X_f[1]) # FFT
                    yf = yf/np.sqrt(N) #Normalize FFT
                    xf = fftfreq(N, T) # FFT frequencies 
                    xf = fftshift(xf) #FFT shift
                    yplot = fftshift(abs(yf))
                    plt.figure(1)
                    plt.gcf().clear()
                    fft_plot = yplot
                    # Find highest peak between 0.75 and 4 Hz 
                    fft_plot[xf<=0.75] = 0 
                    print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm') # Print heart rate
                    plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)]) # Plot FFT
                    plt.pause(0.001)



Some notes about the code:

1. Make sure to put the location of your cascade classifier in the code

2. While measuring, keep still as it only detects the region of interest in the first frame


Here are some figures of the signals as requested by @A_A. This data was obtained with the same program but with a frame rate of 20 fps. However, the data and result is similar for 30fps signal as well.

Normalized RGB signals (color is the signal from that color channel):

[![Normalized RGB signal][2]][2]

After the RGB signals are passed into the ICA, the components (filtered with a 4th order Butterworth bandpass):

[![First Component][3]][3]
[![Second Component][4]][4]
[![Third Component][5]][5]

Finally, here is the FFT of the signal. The maximum peak is what is marked as the heart rate. Note that lower frequencies below 0.75 Hz (45 bpm) are already discarded.

[![FFT][6]][6]

I did a plot with the no face and just background. The peak in the RGB signals might be because I accidentally came into the field of view.

[![RGB - background][7]][7] 
[![First Component - Background][8]][8]
[![Second Component - Background][9]][9]
[![Third Component - Background][10]][10]
[![FFT - Background][11]][11]


  [1]: https://www.osapublishing.org/oe/abstract.cfm?URI=oe-18-10-10762
  [2]: https://i.stack.imgur.com/hkE7l.png
  [3]: https://i.stack.imgur.com/v6HsQ.png
  [4]: https://i.stack.imgur.com/Vc7Pb.png
  [5]: https://i.stack.imgur.com/Tj3Hn.png
  [6]: https://i.stack.imgur.com/JKnrV.png
  [7]: https://i.stack.imgur.com/fGR5C.png
  [8]: https://i.stack.imgur.com/VFfia.png
  [9]: https://i.stack.imgur.com/j35s2.png
  [10]: https://i.stack.imgur.com/ZdJly.png
  [11]: https://i.stack.imgur.com/qzfMH.png
