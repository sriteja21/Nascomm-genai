import cv2
import mediapipe as mp
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
import os
import time
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pyautogui

# -------------------------------
# 1. GOOGLE GENERATIVE AI SETUP
# -------------------------------
load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=my_api_key)

gesture_cache = {}

# Predefined AI assistant responses
custom_responses = {
    "Thumbs Up": "Great! You agreed. 👍",
    "Waving": "Hello! Nice to see you waving! 👋",
    "OK Sign": "Got it! Everything seems good. 😊",
    "Fist": "That looks powerful! What's on your mind? 💪",
    "Victory/Peace": "Peace! Hope you're having a great day. ✌️",
}

def generate_narrative(gesture):
    """Generate an AI assistant response based on the recognized gesture."""
    
    if gesture in ["Unknown Gesture", "No hand detected"]:
        return gesture  # Return as-is for undefined gestures

    # Return custom response if gesture is predefined
    if gesture in custom_responses:
        return custom_responses[gesture]

    # Return cached response if available
    if gesture in gesture_cache:
        return gesture_cache[gesture]

    model = genai.GenerativeModel("gemini-1.5-pro")

    # Retry logic for quota exhaustion
    max_retries = 3
    wait_time = 2  # Initial wait time in seconds

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                f"You are an AI assistant. The user just made a '{gesture}' hand gesture. Respond in a friendly and engaging way."
            )

            # Check if response is valid and contains text
            if response and hasattr(response, "text") and response.text.strip():
                ai_response = response.text.strip()
                gesture_cache[gesture] = ai_response  # Cache response
                return ai_response
            else:
                return "Hmm, I'm not sure how to respond to that gesture."

        except ResourceExhausted:
            print(f"Quota exhausted. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Oops! Something went wrong."

    return "API quota exceeded. Try again later."

# -------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points.
       p1, p2 can be in the form [id, x, y] or (x, y)."""
    if len(p1) == 3:
        x1, y1 = p1[1], p1[2]
    else:
        x1, y1 = p1
    if len(p2) == 3:
        x2, y2 = p2[1], p2[2]
    else:
        x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_wrist_to_index(lmList):
    """Compute the angle (in degrees) from the wrist (landmark 0) 
       to the index fingertip (landmark 8)."""
    wrist_x, wrist_y = lmList[0][1], lmList[0][2]
    index_x, index_y = lmList[8][1], lmList[8][2]
    angle = math.degrees(math.atan2(index_y - wrist_y, index_x - wrist_x))
    return angle

def pinch_distance(lmList):
    """Distance between thumb tip (landmark 4) and index tip (landmark 8)."""
    return euclidean_distance(lmList[4], lmList[8])

def hand_center(lmList):
    """Compute the approximate center of the hand by averaging landmark positions."""
    xs = [pt[1] for pt in lmList]
    ys = [pt[2] for pt in lmList]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

# -------------------------------
# 3. STATIC GESTURE RECOGNITION
# -------------------------------
def recognize_static_gesture(lmList):
    """Recognize static gestures based on fingertip vs. PIP positions."""
    if not lmList:
        return "No hand detected"
    
    # Fingertip y-coordinates
    thumb_tip = lmList[4][2]
    index_tip = lmList[8][2]
    middle_tip = lmList[12][2]
    ring_tip = lmList[16][2]
    pinky_tip = lmList[20][2]
    
    # PIP joint y-coordinates
    index_pip = lmList[6][2]
    middle_pip = lmList[10][2]
    ring_pip = lmList[14][2]
    pinky_pip = lmList[18][2]
    
    ok_threshold = 30  # threshold for OK sign pinch
    
    # OK Sign: Thumb and index tips close, others extended
    if euclidean_distance(lmList[4], lmList[8]) < ok_threshold and \
       middle_tip < middle_pip and ring_tip < ring_pip and pinky_tip < pinky_pip:
        return "OK Sign"
    
    # Thumbs Up: Thumb extended while others are folded
    if thumb_tip < index_pip and \
       middle_tip > index_pip and ring_tip > index_pip and pinky_tip > index_pip:
        return "Thumbs Up"
    
    # Open Palm: All fingers extended
    if index_tip < index_pip and middle_tip < middle_pip and \
       ring_tip < ring_pip and pinky_tip < pinky_pip:
        return "Open Palm"
    
    # Fist: All fingers folded
    if index_tip > index_pip and middle_tip > middle_pip and \
       ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Fist"
    
    # Pointing: Only index finger extended
    if index_tip < index_pip and \
       middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Pointing"
    
    # Victory/Peace: Index and middle extended; ring and pinky folded
    if index_tip < index_pip and middle_tip < middle_pip and \
       ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Victory/Peace"
    
    # Rock Sign: Index and pinky extended; middle and ring folded
    if index_tip < index_pip and \
       middle_tip > middle_pip and ring_tip > ring_pip and \
       pinky_tip < pinky_pip:
        return "Rock Sign"
    
    # Shaka: Thumb and pinky extended; others folded
    if thumb_tip < lmList[3][2] and \
       index_tip > index_pip and middle_tip > middle_pip and ring_tip > ring_pip and \
       pinky_tip < pinky_pip:
        return "Shaka"
    
    return "Unknown Gesture"

# -------------------------------
# 4. DYNAMIC GESTURE RECOGNITION
# -------------------------------

def recognize_dynamic_gesture(hand_id, lmList, handHistory, currentTime):
    dynamic_gesture = None
    center_now = hand_center(lmList)
    
    if hand_id not in handHistory:
        handHistory[hand_id] = {
            'center': center_now,
            'lastGestureTime': 0,
            'lastGesture': None
        }
        return None
    
    center_prev = handHistory[hand_id]['center']
    dx = center_now[0] - center_prev[0]
    swipe_threshold = 80
    cooldown_time = 1.0
    last_time = handHistory[hand_id]['lastGestureTime']



    if (currentTime - last_time) > cooldown_time:
        if dx > swipe_threshold:
            dynamic_gesture = "Swipe Right"
        elif dx < -swipe_threshold:
            dynamic_gesture = "Swipe Left"

        if dynamic_gesture:
            handHistory[hand_id]['lastGestureTime'] = currentTime
            handHistory[hand_id]['center'] = center_now
            return dynamic_gesture
    
    handHistory[hand_id]['center'] = center_now  # Update center even if no gesture
    return None
class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComp,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    @staticmethod
    def draw_multiline_text(img, text, position, font, scale, color, thickness, line_spacing=30):
        x, y = position
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (x, y + i * line_spacing), font, scale, color, thickness)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, draw=True):
        """Return a list of landmark lists (one per detected hand)."""
        allHands = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                allHands.append(lmList)
        return allHands

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = handDetector()
        self.handHistory = {}
        self.last_static_gesture = None  # Store last detected static gesture

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.detector.findHands(img)
        lmLists = self.detector.findPositions(img)

        if lmLists:
            gesture = recognize_static_gesture(lmLists[0])
            cv2.putText(img, f"Gesture: {gesture}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if gesture != self.last_static_gesture:
                if gesture == "Open Palm":
                    pyautogui.scroll(500)
                elif gesture == "Fist":
                    pyautogui.scroll(-500)
            self.last_static_gesture = gesture
            res = generate_narrative(gesture)
            cv2.putText(img, res, (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Check dynamic gesture regardless of finger count
            currentTime = time.time()
            dynamic_gesture = recognize_dynamic_gesture(0, lmLists[0], self.handHistory, currentTime)
            if dynamic_gesture:
                cv2.putText(img, f"Dynamic: {dynamic_gesture}", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if dynamic_gesture == "Swipe Right":
                    pyautogui.hotkey("ctrl", "tab")
                elif dynamic_gesture == "Swipe Left":
                    pyautogui.hotkey("ctrl", "shift", "tab")
        return img

st.title("🤖 Hand Gesture Recognition with AI ✋")
st.write("This app detects hand gestures in real-time using *MediaPipe* and describes them using *Google Gemini AI*.")
webrtc_streamer(key="gesture-detection", video_transformer_factory=VideoTransformer)