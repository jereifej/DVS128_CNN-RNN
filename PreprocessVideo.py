import cv2
import numpy as np
import os

# This was a bad idea, don't use this #
path = 'dataset/wave/test'

l_files = os.listdir(path)

postidx = 1
for file in l_files:
    filename = path + '/' + file.title()
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Error opening video file")
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    acc = np.zeros_like(frame, dtype=int)
    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            acc = acc + np.array(frame, dtype=int)
            frame_count = frame_count + 1
            cv2.imshow(file.title(), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    avg = acc / frame_count
    avg = np.array(avg, dtype=np.uint8)  # subtract current and previous frame, then threshold
    cv2.imwrite(path + '/post' + str(postidx) + '.jpg', avg)

    postidx = postidx + 1
    cap.release()
    cv2.destroyAllWindows()
