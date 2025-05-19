import time
import cv2
import mediapipe as mp
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture('PoseVideos/3.mp4')
pTime = cTime = 0
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    img = cv2.resize(img, (400, 600))
    if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cv = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cv)
                # if id==25:
                cv2.circle(img,(cx,cv),10,(255,0,0),cv2.FILLED)
            mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Image", 400, 600)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
