#  python Real_Time_Obejct_Detection_System.py -p Model/MobileNetSSD_deploy.prototxt.txt -m Model/MobileNetSSD_deploy.caffemodel
# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Construct the argument parse and parse the arguments
# Arguments used here:
# prototxt = MobileNetSSD_deploy.prototxt.txt (required)
# model = MobileNetSSD_deploy.caffemodel (required)
# confidence = 0.2 (default)
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

#  Initializing labels and colors with object names and assigning random color to each label
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "table",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "Laptop", ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#  Loading caffe Model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#  Initializing video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)         # warming up the camera for few  seconds
fps = FPS().start()

#  Loop video Stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    print(frame.shape)
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, (1 / 127.5), (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    for i in np.arange(0, predictions.shape[2]):

        confidence = predictions[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(predictions[0, 0, i, 1])
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("Object detected: ", label)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

# Press 's' key to break the loop
    if key == ord("s"):
        break
    fps.update()
fps.stop()

# Display FPS Information
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
