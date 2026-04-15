from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import threading
import time

app = Flask(__name__)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

gesture_state = {
    "gesture": "none",
    "cursor_x": 0,
    "cursor_y": 0,
    "fps": 0,
    "hand_detected": False,
}
lock = threading.Lock()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = None
camera_running = False

smooth_x, smooth_y = 0, 0
SMOOTH_FACTOR = 0.35

last_click_time = 0
CLICK_COOLDOWN = 0.5

last_scroll_time = 0
SCROLL_COOLDOWN = 0.1

prev_scroll_y = None
drag_mode = False


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_finger_states(landmarks):
    """Returns [thumb, index, middle, ring, pinky] — True if finger is up."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = []

    # Thumb: compare x position (works for right hand facing camera)
    fingers.append(landmarks[4].x < landmarks[3].x)

    # Other fingers: tip above pip joint
    for tip, pip in zip(tips, pips):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers  # [thumb, index, middle, ring, pinky]


def process_frame(frame):
    global smooth_x, smooth_y, last_click_time, last_scroll_time, prev_scroll_y, drag_mode

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_name = "No Hand"
    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        lm = result.multi_hand_landmarks[0]
        landmarks = lm.landmark

        mp_draw.draw_landmarks(
            frame, lm,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        fingers = get_finger_states(landmarks)
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        wrist = landmarks[0]

        # Smooth cursor from index fingertip
        raw_x = np.interp(index_tip.x, [0.1, 0.9], [0, SCREEN_W])
        raw_y = np.interp(index_tip.y, [0.1, 0.9], [0, SCREEN_H])
        smooth_x += (raw_x - smooth_x) * SMOOTH_FACTOR
        smooth_y += (raw_y - smooth_y) * SMOOTH_FACTOR
        cx, cy = int(smooth_x), int(smooth_y)

        now = time.time()

        # ── GESTURE 1: MOVE ──────────────────────────────────────────
        # Only index finger up, all others down
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            if drag_mode:
                drag_mode = False
                pyautogui.mouseUp()
            gesture_name = "Move"
            pyautogui.moveTo(cx, cy, duration=0)
            prev_scroll_y = None

        # ── GESTURE 2: CLICK ─────────────────────────────────────────
        # Index + middle up, ring + pinky down (✌️ peace sign)
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            if drag_mode:
                drag_mode = False
                pyautogui.mouseUp()
            if (now - last_click_time) > CLICK_COOLDOWN:
                gesture_name = "Click!"
                pyautogui.click(cx, cy)
                last_click_time = now
            else:
                gesture_name = "Click (cooldown)"
            prev_scroll_y = None

        # ── GESTURE 3: SCROLL ────────────────────────────────────────
        # All 5 fingers up (open palm) — move hand up/down to scroll
        elif fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            if drag_mode:
                drag_mode = False
                pyautogui.mouseUp()
            gesture_name = "Scroll"
            mid_y = (index_tip.y + middle_tip.y) / 2
            if prev_scroll_y is not None and (now - last_scroll_time) > SCROLL_COOLDOWN:
                delta = (mid_y - prev_scroll_y) * 1000
                if abs(delta) > 0.3:
                    pyautogui.scroll(int(-delta))
                    last_scroll_time = now
            prev_scroll_y = mid_y

        # ── GESTURE 4: DRAG ──────────────────────────────────────────
        # Fist: all four fingers curled down
        elif not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            if not drag_mode:
                drag_mode = True
                pyautogui.mouseDown()
            gesture_name = "Drag"
            pyautogui.moveTo(cx, cy, duration=0)
            prev_scroll_y = None

        # ── IDLE ─────────────────────────────────────────────────────
        else:
            gesture_name = "Idle"
            if drag_mode:
                drag_mode = False
                pyautogui.mouseUp()
            prev_scroll_y = None

        # Dot on index fingertip
        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)
        cv2.circle(frame, (ix, iy), 10, (0, 255, 120), -1)

        with lock:
            gesture_state["cursor_x"] = cx
            gesture_state["cursor_y"] = cy

    else:
        if drag_mode:
            drag_mode = False
            pyautogui.mouseUp()
        prev_scroll_y = None

    # HUD
    cv2.rectangle(frame, (0, 0), (w, 40), (15, 15, 25), -1)
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 255), 2)

    with lock:
        gesture_state["gesture"] = gesture_name
        gesture_state["hand_detected"] = hand_detected

    return frame


def gen_frames():
    global cap, camera_running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    camera_running = True

    prev_time = time.time()
    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        with lock:
            gesture_state["fps"] = int(fps)

        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 160), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if cap:
        cap.release()


@app.route('/')
def index():
    return render_template('index.html', screen_w=SCREEN_W, screen_h=SCREEN_H)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def state():
    with lock:
        return jsonify(gesture_state)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    print(f"Virtual Mouse | Screen: {SCREEN_W}x{SCREEN_H}")
    print("Open: http://localhost:5000")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)