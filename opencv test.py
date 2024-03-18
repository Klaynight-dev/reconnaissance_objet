from time import perf_counter
import cv2

t1_start = perf_counter()
frame_count = 0
webcam = cv2.VideoCapture(0)
NB_IMAGES = 100

if webcam.isOpened():
    while (frame_count < NB_IMAGES):
        bImgReady, imageframe = webcam.read() # get frame per frame from the webcam
        frame_count += 1
        
    t1_stop = perf_counter()
    print ("Frame per Sec.: ", NB_IMAGES / (t1_stop - t1_start))

    webcam.release()
    cv2.destroyAllWindows()
    
classCascadefacial = cv2.CascadeClassifier(r'C:\chemin\vers\haarcascade_frontalface_default.xml')

def facialDetectionAndMark(_image, _classCascade):
    imgreturn = _image.copy()
    gray = cv2.cvtColor(imgreturn, cv2.COLOR_BGR2GRAY)
    faces = _classCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(imgreturn, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return imgreturn

def videoDetection(_haarclass):
    webcam = cv2.VideoCapture(0)
    if webcam.isOpened():
        while True:
            bImgReady, imageframe = webcam.read() # get frame per frame from the webcam
            if bImgReady:
                face = facialDetectionAndMark(imageframe, _haarclass)
                cv.imshow('My webcam', face) # show the frame
            else:
                print('No image available')
            keystroke = cv2.waitKey(20) # Wait for Key press
            if (keystroke == 27):
                break # if key pressed is ESC then escape the loop

        webcam.release()
        cv2.destroyAllWindows()   
        
videoDetection(classCascadefacial)

