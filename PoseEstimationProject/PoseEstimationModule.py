import time
import cv2
import mediapipe as mp
import math

class poseDetector():
    def __init__(self,complexity = 1,enableSeg = False,smoothSeg = True,mode=False,upBody = False,smooth = True,detectionCon=0.5,trackCon = 0.5):
        self.mode = mode
        # self.upBody = upBody
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.enableSeg,self.smoothSeg, self.smooth,self.detectionCon,self.trackCon)
    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        img = cv2.resize(img, (400, 600))
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    # def findAngle(self,img,p1,p2,p3,draw=True):
    #     x1, y1 = self.lmList[p1][1:]
    #     x2, y2 = self.lmList[p2][1:]
    #     x3, y3 = self.lmList[p3][1:]
    #     angle = math.degrees(math.atan(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
    #     if angle<0:
    #         angle+=360


def main():
    pTime= cTime = 0
    cap = cv2.VideoCapture('PoseVideos/3.mp4')
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw = False)
        if len(lmList)!=0:
            print(lmList[14])
            cv2.circle(img,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()
