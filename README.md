# reconnaissance_objet
https://maker.pro/raspberry-pi/tutorial/how-to-create-object-detection-with-opencv
```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

r, h, c, w = 200, 20, 300, 20  # simply hardcoded the values
track_window = (c, r, w, h)

while True:
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) != 0:
largest_contour = max(contours, key = cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
ret, track_window = cv2.meanShift(thresh, track_window, term_crit)

x, y, w, h = track_window
cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Object Tracking', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()
```
