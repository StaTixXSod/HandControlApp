import time
import numpy as np
import cv2
import mediapipe as mp
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
import pyautogui as pg
import onnxruntime

en = time.time()

mouse = MouseController()


class HandControlModule:
    def __init__(
            self,
            mode=0,
            num_hands=1,
            det_conf=0.3,
            track_conf=0.3,
            window=5,
            cursor_velocity=200,
            scrolling_velocity=100,
            click_distance=0.05,
            swipe_distance=0.15,
            swipe_direction="horizontal"
    ) -> None:

        # PARAMETERS
        self.window = window
        self.cursor_velocity = cursor_velocity
        self.scrolling_velocity = scrolling_velocity
        self.click_distance = click_distance
        self.swipe_distance = swipe_distance
        self.swipe_direction = swipe_direction

        # MediaPipe Hands Settings
        self.results = None
        self.hands_solution = mp.solutions.hands
        self.hands = self.hands_solution.Hands(
            mode,
            max_num_hands=num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        self.draw_styles = mp.solutions.drawing_styles
        self.draw_utils = mp.solutions.drawing_utils

        # Load model
        self.ort_session = onnxruntime.InferenceSession("models/HPD/HandPoseDetectorOnnx.onnx")

        # Initialize Hand Positions
        self.landmarks = None

        self.poses = ["nothing", "swiping", "scrolling", "pointing"]
        self.pointer_points = [8]
        self.scrolling_points = [8, 12]
        self.swiping_points = [8, 12, 16, 20]

        self.swiping_positions = []
        self.scrolling_positions = {"p_pos": 0.0, "diff": []}
        self.cursor_positions = {"p_X": 0.0, "p_Y": 0.0, "diff_X": [], "diff_Y": [], "acc_X": [], "acc_Y": []}

        self.click = False

    def find_hand(self, image, draw=True):
        """
        Run MediaPipe Hands solution on image.
        Method returns image with hand landmarks on it,
        besides it saves results of founded landmarks for future.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if draw:
            if self.results.multi_hand_landmarks:
                for lms in self.results.multi_hand_landmarks:
                    self.draw_utils.draw_landmarks(
                        image, lms, self.hands_solution.HAND_CONNECTIONS,
                        self.draw_styles.get_default_hand_landmarks_style(),
                        self.draw_styles.get_default_hand_connections_style()
                    )
        return image

    def get_landmarks(self):
        """
        Method returns landmarks of detected hand from the result we calculated above
        """
        landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]

            for lm in hand.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

        self.landmarks = np.array(landmarks)
        return np.array(landmarks)

    def predict_label(self, landmarks):
        """
        Predict label using model with 2 conv layers and 3 FC.
        Method preprocess the data and return gesture label.
        """
        if len(landmarks) == 0:
            self.nullify_values()
            return self.poses[0]

        landmarks = landmarks.flatten().astype(np.float32).reshape(1, 63)
        ort_inputs = {self.ort_session.get_inputs()[0].name: landmarks}
        ort_outs = self.ort_session.run(None, ort_inputs)
        answer = ort_outs[0]
        label = self.poses[np.argmax(answer)]
        return label

    def handle_gesture(self, image, gesture):
        if gesture == "swiping":
            draw_points(image, self.landmarks, self.swiping_points)
            draw_mid_point(image, self.landmarks, self.swiping_points)
            self.swiping_mode(self.landmarks)

        if gesture == "scrolling":
            draw_points(image, self.landmarks, self.scrolling_points)
            draw_mid_point(image, self.landmarks, self.scrolling_points)
            self.scrolling_mode_momentum(self.landmarks)

        if gesture == "pointing":
            draw_points(image, self.landmarks, self.pointer_points)
            draw_mid_point(image, self.landmarks, self.pointer_points)
            self.cursor_mode_momentum(self.landmarks)

        if gesture == "nothing":
            # NULL ALL LISTS
            self.nullify_values()

    def nullify_values(self):
        self.swiping_positions = []
        self.scrolling_positions = {"p_pos": 0.0, "diff": []}
        self.cursor_positions = {"p_X": 0.0, "p_Y": 0.0, "diff_X": [], "diff_Y": [], "acc_X": [], "acc_Y": []}

    def scrolling_mode_momentum(self, landmarks):
        """
        Perform scrolling with Momentum method.
        """
        scrolling_pos = landmarks[self.scrolling_points]
        current_pos = find_mid_point(scrolling_pos)[1]

        if self.scrolling_positions['p_pos'] == 0.0:
            self.scrolling_positions['p_pos'] = current_pos

        if len(self.scrolling_positions['diff']) < self.window:
            diff = current_pos - self.scrolling_positions['p_pos']
            self.scrolling_positions['p_pos'] = current_pos
            self.scrolling_positions['diff'].append(diff)

        if len(self.scrolling_positions['diff']) >= self.window:
            diff = current_pos - self.scrolling_positions['p_pos']
            self.scrolling_positions['p_pos'] = current_pos

            self.scrolling_positions['diff'].append(diff)
            self.scrolling_positions['diff'].pop(0)

            step = sum(self.scrolling_positions['diff'])
            mouse.scroll(0, self.scrolling_velocity * (step / 2))

    def cursor_mode_momentum(self, landmarks):
        """
        Cursor mode. Allows move cursor with hand
        """
        cursor_pose = landmarks[self.pointer_points]
        current_x, current_y, current_z = cursor_pose[0][0], cursor_pose[0][1], cursor_pose[0][2]

        if self.cursor_positions['p_X'] == 0.0 and self.cursor_positions['p_Y'] == 0.0:
            self.cursor_positions['p_X'] = current_x
            self.cursor_positions['p_Y'] = current_y

        if len(self.cursor_positions['diff_X']) < self.window:
            diff_x = current_x - self.cursor_positions['p_X']
            diff_y = current_y - self.cursor_positions['p_Y']

            self.cursor_positions['p_X'] = current_x
            self.cursor_positions['p_Y'] = current_y

            self.cursor_positions['diff_X'].append(diff_x)
            self.cursor_positions['diff_Y'].append(diff_y)

        else:
            diff_x = current_x - self.cursor_positions['p_X']
            diff_y = current_y - self.cursor_positions['p_Y']

            self.cursor_positions['p_X'] = current_x
            self.cursor_positions['p_Y'] = current_y

            self.cursor_positions['diff_X'].append(diff_x)
            self.cursor_positions['diff_Y'].append(diff_y)

            self.cursor_positions['diff_X'].pop(0)
            self.cursor_positions['diff_Y'].pop(0)

            step_x = sum(self.cursor_positions['diff_X'])
            step_y = sum(self.cursor_positions['diff_Y'])

            mouse.move(-self.cursor_velocity * step_x, self.cursor_velocity * step_y)

        click_pose = self.landmarks[[4, 10]]
        thumb_x, thumb_y, thumb_z = click_pose[0][0], click_pose[0][1], click_pose[0][2]
        middle_x, middle_y, middle_z = click_pose[1][0], click_pose[1][1], click_pose[1][2]

        click_distance = get_distance([thumb_x, thumb_y, thumb_z], [middle_x, middle_y, middle_z])

        # If click distance less than 0.03, set click as True for making just one click
        if click_distance < self.click_distance and self.click is False:
            self.click = True
            mouse.click(Button.left)

        # If click distance again or just greater 0.03
        elif click_distance >= self.click_distance:
            self.click = False

    def swiping_mode(self, landmarks):
        """
        If swipe mode is ON, this method perform swiping between desktops
        """
        swipe_pose = landmarks[self.swiping_points]
        mid_point = find_mid_point(swipe_pose)

        current_x, current_y = mid_point[0], mid_point[1]

        if self.swipe_direction == "vertical":
            if len(self.swiping_positions) == 0:
                self.swiping_positions.append(current_y)

            else:
                self.swiping_positions.append(current_y)
                diff = self.swiping_positions[-1] - self.swiping_positions[0]

                if abs(diff) > self.swipe_distance and diff > 0:
                    print("UP")
                    pg.hotkey("ctrl", "alt", "up")
                    self.swiping_positions = []

                if abs(diff) > self.swipe_distance and diff < 0:
                    print("DOWN")
                    pg.hotkey("ctrl", "alt", "down")
                    self.swiping_positions = []

                if len(self.swiping_positions) > self.window:
                    self.swiping_positions.pop(0)

        if self.swipe_direction == "horizontal":
            if len(self.swiping_positions) == 0:
                self.swiping_positions.append(current_x)

            else:
                self.swiping_positions.append(current_x)
                diff = self.swiping_positions[-1] - self.swiping_positions[0]

                if abs(diff) > self.swipe_distance:
                    if diff > 0:
                        pg.hotkey("ctrl", "right")
                        self.swiping_positions = []

                    elif diff < 0:
                        pg.hotkey("ctrl", "left")
                        self.swiping_positions = []

                if len(self.swiping_positions) > self.window:
                    self.swiping_positions.pop(0)


def draw_points(image, landmarks, points2draw):
    """
    Method draw chosen landmarks on image.
    points2draw -> some list, like [8, 12]
    """
    h, w, c = image.shape
    points = landmarks[points2draw]

    for lm in points:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (cx, cy), 10, (255, 255, 255), 2, cv2.FILLED)


def get_distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    distance = (((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2)) ** 0.5
    return distance


def find_mid_point(points):
    """
    Method calculates mean for X, Y, Z points separately
    """
    length = len(points)
    x_mid, y_mid, z_mid = 0, 0, 0
    for point in points:
        x, y, z = point
        x_mid += x
        y_mid += y
        z_mid += z

    return x_mid / length, y_mid / length, z_mid / length


def draw_mid_point(image, landmarks, points2draw):
    """
    Method draws midpoint of chosen landmarks on image.
    points2draw -> some list, like [8, 12]
    """
    h, w, c = image.shape
    points = landmarks[points2draw]
    midpoint = find_mid_point(points)
    cx, cy = int(midpoint[0] * w), int(midpoint[1] * h)

    cv2.circle(image, (cx, cy), 10, (0, 255, 255), 2, cv2.FILLED)
