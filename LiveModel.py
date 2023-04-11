import cv2
import numpy as np
import keras.models
from skimage.util import random_noise


def threshold(frame):
    out = np.full_like(frame, 128)  # start and assume there are no events
    out[frame < -10] = 0  # if event is negative enough, OFF event
    out[frame > 10] = 255  # if event is positive enough, ON event
    return out


model = keras.models.load_model("CNN_model")
cap = cv2.VideoCapture(1)

_, fc = cap.read()
fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
fp = np.array(fc, dtype=int)
while cap.isOpened():
    _, fc = cap.read()
    if type(fc) is not np.ndarray:
        break

    # Do the DVS Camera Emulation
    fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    fc = cv2.GaussianBlur(fc, (9, 9), 20)  # blur bc my webcam is *fart noises*
    fc = np.array(fc, dtype=int)
    diff = fc - fp
    ft = np.array(threshold(diff), dtype=np.uint8)  # subtract current and previous frame, then threshold

    # Setting image to the input size
    im_in = cv2.resize(ft,
                       (0, 0),
                       fx=.3,
                       fy=.3,
                       interpolation=cv2.INTER_NEAREST)
    im_in = random_noise(im_in, mode='s&p', amount=0.011)

    # predict ignores the first dimension? so I add a dimension in that position to make it happy
    im_in = im_in.reshape((1, im_in.shape[0], im_in.shape[1], 1))
    decision = np.argmax(model.predict(im_in, verbose=0))  # find index of max probability
    text = ""
    if decision == 0:
        text = "down"
    elif decision == 1:
        text = "up"
    elif decision == 2:
        text = "wave"
    cv2.putText(ft, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('output', ft)
    fp = fc

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
