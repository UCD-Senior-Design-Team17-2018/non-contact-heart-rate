I am trying to develop a RGB camera-based non-contact measurement of heart rate using principles of remote photoplethysmography. I want to be able to measure a range of heart rates I have been looking into multiple methods, which are all quite similar though. Typically, they all detect/track the face, and then they spatially average the colors in the facial region. They either do a FFT of the green or red temporal signal, or they do the FFT of a component coming from a blind-source separation method (like ICA or PCA). Now, I have been working on this, however I have noticed that the frequency seems to always stay around 50bpm (my heart rate is usually around 70bpm), occasionally jumping to the correct bpm. I have even checked this with a fitness tracker. I have tried adding filters, detrending, etc. but with no success. What could be the problem here? Could there be some sort of other frequencies I am not considering here from the environment? I have tried this out in various environments, and illuminations and that does not seem to affect the problem. Is FFT not robust enough for this method? I have tried Welch's PSD, which seems like it gives better results, but still has some bias towards 50 bpm (0.83 Hz). 

Here is some code based on [this][1] paper:

    import cv2
    import datetime
    import numpy as np
    from scipy import signal
    from scipy.fftpack import fft, fftfreq, fftshift
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    # Change these variables based on the location of your cascade classifier
    PATH_TO_HAAR_CASCADES = "..." 
    face_cascade = cv2.CascadeClassifier(PATH_TO_HAAR_CASCADES+'haarcascade_frontalface_default.xml') # Full pathway must be used
    firstFrame = None
    time = []
    R = []
    G = []
    B = []
    pca = FastICA(n_components=3)
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("Failed to open webcam")
    frame_num = 0
    plt.ion()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame_num += 1
            if firstFrame is None:
                start = datetime.datetime.now()
                time.append(0)
                # Take first frame and find face in it
                firstFrame = frame
                cv2.imshow("frame",firstFrame)
                old_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(old_gray, 1.3, 5) 
                if faces == ():
                    firstFrame = None
                else:
                    for (x,y,w,h) in faces: 
                        x2 = x+w
                        y2 = y+h
                        cv2.rectangle(firstFrame,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.imshow("frame",firstFrame)
                        VJ_mask = np.zeros_like(firstFrame)
                        VJ_mask = cv2.rectangle(VJ_mask,(x,y),(x+w,y+h),(255,0,0),-1)
                        VJ_mask = cv2.cvtColor(VJ_mask, cv2.COLOR_BGR2GRAY)
                    ROI = VJ_mask
                    ROI_color = cv2.bitwise_and(ROI,ROI,mask=VJ_mask)
                    cv2.imshow('ROI',ROI_color)
                    R_new,G_new,B_new,_ = cv2.mean(ROI_color,mask=ROI)
                    R.append(R_new)
                    G.append(G_new)
                    B.append(B_new)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            else:
                current = datetime.datetime.now()-start
                current = current.total_seconds()
                time.append(current)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ROI_color = cv2.bitwise_and(frame,frame,mask=ROI)
                cv2.imshow('ROI',ROI_color)
                R_new,G_new,B_new,_ = cv2.mean(ROI_color, mask=ROI)
                R.append(R_new)
                G.append(G_new)
                B.append(B_new)
                if frame_num >= 900:
                    N = 900
                    G_std = StandardScaler().fit_transform(np.array(G[-(N-1):]).reshape(-1, 1))
                    G_std = G_std.reshape(1, -1)[0]
                    R_std = StandardScaler().fit_transform(np.array(R[-(N-1):]).reshape(-1, 1))
                    R_std = R_std.reshape(1, -1)[0]
                    B_std = StandardScaler().fit_transform(np.array(B[-(N-1):]).reshape(-1, 1))
                    B_std = B_std.reshape(1, -1)[0]
                    T = 1/(len(time[-(N-1):])/(time[-1]-time[-(N-1)]))
                    X_f=pca.fit_transform(np.array([R_std,G_std,B_std]).transpose()).transpose()
                    N = len(X_f[0])
                    yf = fft(X_f[0])
                    xf = fftfreq(N, T)
                    xf = fftshift(xf)
                    yplot = fftshift(abs(yf))
                    plt.figure(1)
                    plt.gcf().clear()
                    fft_plot = yplot
                    fft_plot[xf<=0.75] = 0
                    print(str(xf[fft_plot[xf<=4].argmax()]*60)+' bpm')
                    plt.plot(xf[(xf>=0) & (xf<=4)], fft_plot[(xf>=0) & (xf<=4)])
                    plt.pause(0.001)



Some notes about the code:

1. Make sure to put the location of your cascade classifier in the code

2. While measuring, keep still as it only detects the region of interest in the first frame

Here are some figures of the signals. This data was obtained with the same program but with a frame rate of 10fps. However, the data and result is similar for 30fps signal as well.

Normalized RGB signals:

[![Normalized R temporal signal][2]][2]
[![Normalized G temporal signal][3]][3]
[![Normalized B temporal signal][4]][4]

After the RGB signals are passed into the ICA, the components (filtered with a 4th order Butterworth bandpass):

[![First Component][5]][5]
[![Second Component][6]][6]
[![Third Component][7]][7]

Finally, here is the FFT of the signal. The maximum peak is what is marked as the heart rate. Note that lower frequencies below 0.75 Hz (45 bpm) are already discarded.

[![FFT][8]][8]


  [1]: https://www.osapublishing.org/oe/abstract.cfm?URI=oe-18-10-10762
  [2]: https://i.stack.imgur.com/Jj0SN.png
  [3]: https://i.stack.imgur.com/Fswne.png
  [4]: https://i.stack.imgur.com/Xb07O.png
  [5]: https://i.stack.imgur.com/8jASv.png
  [6]: https://i.stack.imgur.com/S3u72.png
  [7]: https://i.stack.imgur.com/3BiLn.png
  [8]: https://i.stack.imgur.com/4qzk5.png
