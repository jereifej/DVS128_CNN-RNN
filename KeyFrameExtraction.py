# Saves a set of Key Frames for training

import cv2
import numpy as np
import os


def KeyFrameExtraction(source, target, target_filename):
    # get source video
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    threshold_min = 6000
    threshold_max = 12000
    framecount = 0
    while cap.isOpened():
        _, fc = cap.read()
        if type(fc) is not np.ndarray:
            break
        name = target_directory + "/" + target_filename + "_" + str(framecount)
        print(name)
        fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        acc = 0
        flat = np.reshape(fc, (np.product(fc.shape),))
        acc = np.sum(np.where(flat == 0, 1, 0)) + np.sum(np.where(flat == 255, 1, 0))
        print(acc)
        if threshold_min < acc < threshold_max:
            cv2.imwrite(name + '.jpg', fc)

        cv2.imshow(source, fc)
        framecount = framecount + 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


source_directory = "dataset/thumbsdown/train/"
target_directory = "dataset/thumbsdown/KeyFrames"

# file_count = 1
# for filename in os.listdir(source_directory):
#     print(filename)
#     source_file = source_directory + "/" + filename
#     KeyFrameExtraction(source=source_file, target=target_directory, target_filename="wave"+str(file_count))
#     file_count = file_count + 1

KeyFrameExtraction(source="dataset/thumbsdown/video/down4.mp4",
                   target=target_directory, target_filename="down4")