import cv2
import numpy as np
import keras.models
import time
from skimage.util import random_noise


def threshold(frame):
    out = np.full_like(frame, 128)  # start and assume there are no events
    out[frame < -10] = 0  # if event is negative enough, OFF event
    out[frame > 10] = 255  # if event is positive enough, ON event
    return out


# set up model and OpenCV video capture settings
model = keras.models.load_model("CNN_model")
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FPS, int(30))


# initialize current and previous frame
_, fc = cap.read()
fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
fp = np.array(fc, dtype=int)
print(np.shape(fc))
while cap.isOpened():
    _, fc = cap.read()
    if type(fc) is not np.ndarray:
        break
    start = time.time()
    # do the DVS camera emulation thing
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

    print(str((time.time()-start)*1E3) + " ms")
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
