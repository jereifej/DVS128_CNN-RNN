import cv2
import numpy as np
import os

def threshold(frame):
    out = np.full_like(frame, 128)  # start and assume there are no events
    out[frame < -10] = 0  # if event is negative enough, OFF event
    out[frame > 10] = 255  # if event is positive enough, ON event
    return out

def ProcessVideo(filename, target):
    # get source video
    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # define target directory
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(target, fourcc, 24, (width, height))

    ret, fc = cap.read()
    fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
    fp = np.array(fc, dtype=int)
    while cap.isOpened():
        if ret:
            _, fc = cap.read()
            if type(fc) is not np.ndarray:
                break

            fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            fc = cv2.GaussianBlur(fc, (9, 9), 20)  # blur bc my webcam is *fart noises*
            fc = np.array(fc, dtype=int)
            diff = fc - fp
            ft = np.array(threshold(diff), dtype=np.uint8)  # subtract current and previous frame, then threshold
            ft = cv2.cvtColor(ft, cv2.COLOR_GRAY2RGB)
            cv2.imshow('output', ft)
            writer.write(ft)
            # print(diff)
            fp = fc


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# source_directory = "dataset/input/thumbsdown"
# target_directory = "dataset/thumbsdown/train"
# for filename in os.listdir(source_directory):
#     print(filename)
#     source = source_directory + "/" + filename
#     target = target_directory + "/" + filename
#     ProcessVideo(filename=source, target=target)
ProcessVideo(filename="downTEST.mp4", target="downout.mp4")



