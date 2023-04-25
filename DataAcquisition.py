import cv2
import numpy as np
import os


# This is the all-inclusive Data Acquisition & Preprocessing File! #

def threshold(frame):
    out = np.full_like(frame, 128)  # start and assume there are no events
    out[frame < -10] = 0  # if event is negative enough, OFF event
    out[frame > 10] = 255  # if event is positive enough, ON event
    return out


def KeyFrameExtraction(frame, threshold_min=170E3, threshold_max=400E3):
    flat = np.reshape(frame, (np.product(frame.shape),))
    acc = np.sum(np.where(flat == 0, 1, 0)) + np.sum(np.where(flat == 255, 1, 0))
    print(acc)
    if threshold_min < acc < threshold_max:
        return True
    return False


def ProcessVideo(filename, target_name, dvs_directory, frame_directory):
    # get source video
    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, fc = cap.read()
    fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
    fp = np.array(fc, dtype=int)

    framecount = 0
    while cap.isOpened():
        if ret:
            _, fc = cap.read()
            if type(fc) is not np.ndarray:
                break
            dvs_name = dvs_directory + "/" + target_name + "_" + str(framecount)
            frame_name = frame_directory + "/" + target_name + "_" + str(framecount)
            print(dvs_name)

            fc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            fc = cv2.GaussianBlur(fc, (9, 9), 20)  # blur bc my webcam is *fart noises*
            fc = np.array(fc, dtype=int)
            diff = fc - fp
            ft = np.array(threshold(diff), dtype=np.uint8)  # subtract current and previous frame, then threshold
            ft = cv2.cvtColor(ft, cv2.COLOR_GRAY2RGB)

            if KeyFrameExtraction(ft):
                print("oh yeah")
                # cv2.imwrite(dvs_name + '.jpg', ft)
                cv2.imwrite(frame_name + '.jpg', fc)
            cv2.imshow('output', ft)
            fp = fc
            framecount = framecount+1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


source_directory = "dataset/input/thumbsdown"
dvs_dir = "dataset/thumbsdown/KeyFrames"
frame_dir = "dataset/real_images/thumbsdown"
for filename in os.listdir(source_directory):
    print(filename)
    source = source_directory + "/" + filename
    dvs_target = dvs_dir
    frame_target = frame_dir
    ProcessVideo(filename=source, dvs_directory=dvs_target, frame_directory=frame_target, target_name=filename)

