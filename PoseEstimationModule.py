import cv2
import mediapipe as mp
import time


class PoseDetector():

    def __init__(self, mode=False, complexity=1, smooth_lms=True, min_detect_conf=0.5, min_track_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lms = smooth_lms
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smooth_lms,
                                      self.min_detect_conf, self.min_track_conf)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        land_marks_list = []
        if self.results.pose_landmarks:
            for id_, land_mark in enumerate(self.results.pose_landmarks.landmark):
                height, width, c = img.shape
                cx, cy = int(land_mark.x * width), int(land_mark.y * height)
                land_marks_list.append([id_, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return land_marks_list


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = PoseDetector()
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

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
