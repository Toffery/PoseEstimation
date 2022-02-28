import cv2
import mediapipe as mp
import time
import PoseEstimationModule as pem


cap = cv2.VideoCapture(0)
prev_time = 0
detector = pem.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    land_marks_list = detector.find_position(img)
    if len(land_marks_list) != 0:
        print(land_marks_list)
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()