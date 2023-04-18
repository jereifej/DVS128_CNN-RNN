import cv2
import numpy as np
import keras.models
import time
from threading import Thread
from skimage.util import random_noise

# Using https://github.com/PyImageSearch/imutils/tree/master/imutils/video
# defining a helper class for implementing multi-threading
class WebcamStream:
    # initialization method
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True


def threshold(frame):
    out = np.full_like(frame, 128)  # start and assume there are no events
    out[frame < -10] = 0  # if event is negative enough, OFF event
    out[frame > 10] = 255  # if event is positive enough, ON event
    return out


model = keras.models.load_model("CNN_model")
webcam_stream = WebcamStream(stream_id=1)
webcam_stream.start()

fc = webcam_stream.read()
fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
fp = np.array(fc, dtype=int)

num_frames_processed = 0
start = time.time()
while True:
    if webcam_stream.stopped is True :
        break
    else:
        fc = webcam_stream.read()

    fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    fc = cv2.GaussianBlur(fc, (9, 9), 20)  # blur bc my webcam is *fart noises*
    fc = np.array(fc, dtype=int)
    diff = fc - fp
    ft = np.array(threshold(diff), dtype=np.uint8)  # subtract current and previous frame, then threshold
    im_in = ft
    # Setting image to the input size
    im_in = cv2.resize(ft,
                       (0, 0),
                       fx=.15,
                       fy=.15,
                       interpolation=cv2.INTER_NEAREST)
    # im_in = random_noise(im_in, mode='s&p', amount=0.011)

    # predict ignores the first dimension? so I add a dimension in that position to make it happy
    im_in = im_in.reshape((1, im_in.shape[0], im_in.shape[1], 1))
    certainty = model.predict(im_in, verbose=0)

    decision = np.argmax(certainty)  # find index of max probability
    text = ""
    # if decision is almost certain...
    if np.max(certainty) > .99:
        if decision == 0:
            text = "thumbs down"
        elif decision == 1:
            text = "thumbs up"
        elif decision == 2:
            text = "wave"
    cv2.putText(ft, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('output', ft)
    fp = fc

    num_frames_processed += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream

# printing time elapsed and fps
elapsed = end-start
fps = num_frames_processed/elapsed
print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
# closing all windows
cv2.destroyAllWindows()