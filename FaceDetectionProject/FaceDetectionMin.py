import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            # print(img.shape)
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
            int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(0,255,0),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
            # mpDraw.draw_detection(img,detection)
    cTime = pTime = 0
    cTime = time.time()
    fps = 1/(cTime-pTime)
    # cv2.putText(img,str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
