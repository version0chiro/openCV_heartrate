import cv2
import imutils
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal



class Webcam(object):
    def __init__(self):
        self.dirname = ""  # for nothing, just to make 2 inputs the same
        self.cap = None

    def start(self):
        print("[INFO] Start webcam")
        time.sleep(1)  # wait for camera to be ready
        self.cap = cv2.VideoCapture(0)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):

        if self.valid:
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            # cv2.putText(frame, str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            #           (65,220), cv2.FONT_HERSHEY_PLAIN, 2, (0,256,256))
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Stop webcam")

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        # self.red = np.zeros((256,256,3),np.uint8)

    def extractColor(self, frame):


        g = np.mean(frame[:, :, 1])
        return g

    def run(self):

        frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)

        self.frame_out = frame
        self.frame_ROI = face_frame

        g1 = self.extractColor(ROI1)
        g2 = self.extractColor(ROI2)

        L = len(self.data_buffer)


        g = (g1 + g2) / 2

        if (abs(g - np.mean(
                self.data_buffer)) > 10 and L > 99):  # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]

        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size // 2:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)

        if L == self.buffer_size:
            self.fps = float(L) / (self.times[-1] - self.times[
                0])  # calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)

            processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed)  # interpolation by 1
            interpolated = np.hamming(
                L) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated / np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm * 30)  # do real fft with the normalization multiplied by 10

            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs


            self.fft = np.abs(raw) ** 2  # get amplitude spectrum

            idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
            pruned = self.fft[idx]
            pfreq = freqs[idx]

            self.freqs = pfreq
            self.fft = pruned

            idx2 = np.argmax(pruned)  # max in the range can be HR

            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)

            processed = self.butter_bandpass_filter(processed, 0.8, 3, self.fps, order=3)

        self.samples = processed  # multiply the signal with 5 for easier to see in the plot


        if (mask.shape[0] != 10):
            out = np.zeros_like(face_frame)
            mask = mask.astype(np.bool)
            out[mask] = face_frame[mask]
            if (processed[-1] > np.mean(processed)):
                out[mask, 2] = 180 + processed[-1] * 10
            face_frame[mask] = out[mask]

    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

process=Process()
inCam=Webcam()
inCam.start()
while 1:

    frame = inCam.get_frame()
    frame = imutils.resize(frame,width=600)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=cv2.flip(frame,1)
    cv2.imshow("test",frame)
    process.frame_in = frame
    process.run()
    cv2.imshow("processed",frame)
    frame=process.frame_out
    f_fr=process.frame_ROI
    bpm=process.bpm
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.putText(frame,"FPS "+str(float("{:.2f}".format(process.fps))),
                (20,450),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,255),2)
    f_fr=np.transpose(f_fr,(0,1,2)).copy()
    print(process.bpm)
    if process.bpms.__len__() > 50:
        if(max(process.bpms-np.mean(process.bpms))<5):
            cv2.putText(frame, "FPS " + str(float("{:.2f}".format(np.mean(process.bpms)))),
                        (20, 250), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    key=cv2.waitKey(1) & 0xFF
    cv2.imshow("output",frame)
    if key == ord('q'):
        print("[info] stop webcam")
        inCam.stop()
        break


