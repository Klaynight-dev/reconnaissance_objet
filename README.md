# reconnaissance_objet
https://maker.pro/raspberry-pi/tutorial/how-to-create-object-detection-with-opencv
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import datetime
from ultralytics import YOLO
import cv2
from imutils.video import VideoStream
#from helper import create_video_writer


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
"""
#detect on image
frame= cv2.imread('./datasets/two-boats.jpg');ret= True #from image file
detections = model(frame)[0]
# loop over the detections
#for data in detections.boxes.data.tolist():
for box in detections.boxes:
    #extract the label name
    label=model.names.get(box.cls.item())
        
    # extract the confidence (i.e., probability) associated with the detection
    data=box.data.tolist()[0]
    confidence = data[4]

    # filter out weak detections by ensuring the
    # confidence is greater than the minimum confidence
    if float(confidence) < CONFIDENCE_THRESHOLD:
        continue

    # if the confidence is greater than the minimum confidence,
    # draw the bounding box on the frame
    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

    #draw confidence and label
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    cv2.putText(frame, "{} {:.1f}%".format(label,float(confidence*100)), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

# show the frame to our screen
cv2.imshow("Img", frame)
"""
#detect on video
# initialize the video capture object
#vs = VideoStream(src=0, resolution=(1600, 1200)).start()
video_cap = cv2.VideoCapture(0)
# initialize the video writer object
#writer = create_video_writer(video_cap, "output.mp4")

while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    
    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]
    
    # loop over the detections
    #for data in detections.boxes.data.tolist():
    for box in detections.boxes:
        #extract the label name
        label=model.names.get(box.cls.item())
        
        # extract the confidence (i.e., probability) associated with the detection
        data=box.data.tolist()[0]
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

        #draw confidence and label
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(frame, "{} {:.1f}%".format(label,float(confidence*100)), (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        #cv2.circle(frame, (int(X)-15, int(Y)), 1, GREEN, 2)
        #cv2.putText(frame, poslbl, (int(X), int(Y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print('Time to procss 1 frame :',round(total * 1000, 0),'miliseconds')

    # calculate the frame per second and draw it on the frame
    fps = 'FPS' + str(round(1/total,2))
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    #writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
vs.stop()
#writer.release()
cv2.destroyAllWindows()

```
