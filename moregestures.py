import cv2
import mediapipe as mp
import google.generativeai as genai
import google.api_core.exceptions  # For ResourceExhausted
from dotenv import load_dotenv
import os
import time
import math

# -------------------------------
# 1. GOOGLE GENERATIVE AI SETUP
# -------------------------------
load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=my_api_key)

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
    """
    Naïvely detect dynamic gestures by comparing the current hand center, angle, and pinch distance
    with those from the previous frame.
    """
    dynamic_gesture = None
    center_now = hand_center(lmList)
    angle_now = angle_wrist_to_index(lmList)
    pinch_now = pinch_distance(lmList)
    
    # If this hand is new, initialize history
    if hand_id not in handHistory:
        handHistory[hand_id] = {
            'center': center_now,
            'angle': angle_now,
            'pinch': pinch_now,
            'lastGestureTime': 0,
            'lastGesture': None
        }
        return None  # not enough info
    
    center_prev = handHistory[hand_id]['center']
    angle_prev = handHistory[hand_id]['angle']
    pinch_prev = handHistory[hand_id]['pinch']
    last_time = handHistory[hand_id]['lastGestureTime']
    
    # Update history for next frame
    handHistory[hand_id]['center'] = center_now
    handHistory[hand_id]['angle'] = angle_now
    handHistory[hand_id]['pinch'] = pinch_now
    
    # Simple cooldown: only allow a dynamic gesture every 1 second per hand
    if (currentTime - last_time) < 1.0:
        return None
    
    dx = center_now[0] - center_prev[0]
    dy = center_now[1] - center_prev[1]
    dAngle = angle_now - angle_prev
    dPinch = pinch_now - pinch_prev
    
    swipe_threshold = 50
    rotate_threshold = 25
    zoom_threshold = 20
    
    # Detect Swipe (choose direction based on larger movement axis)
    if abs(dx) > swipe_threshold or abs(dy) > swipe_threshold:
        if abs(dx) > abs(dy):
            dynamic_gesture = "Swipe Right" if dx > 0 else "Swipe Left"
        else:
            dynamic_gesture = "Swipe Down" if dy > 0 else "Swipe Up"
    # Detect Rotation
    elif abs(dAngle) > rotate_threshold:
        dynamic_gesture = "Rotate Right" if dAngle > 0 else "Rotate Left"
    # Detect Zoom (change in pinch distance)
    elif abs(dPinch) > zoom_threshold:
        dynamic_gesture = "Zoom In" if dPinch < 0 else "Zoom Out"
    
    if dynamic_gesture:
        handHistory[hand_id]['lastGestureTime'] = currentTime
        handHistory[hand_id]['lastGesture'] = dynamic_gesture
        return dynamic_gesture
    
    return None

# -------------------------------
# 5. AI NARRATIVE GENERATION (Optional)
# -------------------------------
# gesture_cache = {}
# def generate_narrative(gesture):
#     """Generate an engaging narrative for a recognized gesture via Google Generative AI."""
#     if gesture in ["Unknown Gesture", "No hand detected", None]:
#         return gesture
#     if gesture in gesture_cache:
#         return gesture_cache[gesture]
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     max_retries = 3
#     wait_time = 2
#     for attempt in range(max_retries):
#         try:
#             prompt = f"Describe the meaning or context of the '{gesture}' gesture in an engaging way."
#             response = model.generate_content(prompt)
#             if response and hasattr(response, "text"):
#                 gesture_cache[gesture] = response.text
#                 return response.text
#             else:
#                 return "Failed to generate description."
#         except google.api_core.exceptions.ResourceExhausted:
#             print(f"Quota exhausted. Retrying in {wait_time} seconds...")
#             time.sleep(wait_time)
#             wait_time *= 2
#         except Exception as e:
#             return f"Error: {str(e)}"
#     return "API quota exceeded. Try again later."

# -------------------------------
# 6. HAND DETECTOR CLASS
# -------------------------------
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

# -------------------------------
# 7. MAIN FUNCTION
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    handHistory = {}  # For dynamic gesture tracking (per hand index)
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img, draw=True)
        handsLM = detector.findPositions(img, draw=False)
        recognized_gestures = []  # Collect gestures for display
        
        # Process each detected hand
        for i, lmList in enumerate(handsLM):
            static_gesture = recognize_static_gesture(lmList)
            currentTime = time.time()
            dynamic_gesture = recognize_dynamic_gesture(i, lmList, handHistory, currentTime)
            # Use dynamic gesture if detected, else static gesture
            gesture = dynamic_gesture if dynamic_gesture else static_gesture
            recognized_gestures.append(gesture)
        
        # Two-Hand Gestures: Namaste and Clapping
        if len(handsLM) == 2:
            center1 = hand_center(handsLM[0])
            center2 = hand_center(handsLM[1])
            dist = euclidean_distance(center1, center2)
            clap_threshold = 100      # Tune this value for clapping detection
            namaste_threshold = 120   # Tune this value for namaste detection

            # Check static gestures for both hands
            gesture1 = recognize_static_gesture(handsLM[0])
            gesture2 = recognize_static_gesture(handsLM[1])
            
            if gesture1 == "Open Palm" and gesture2 == "Open Palm" and dist < namaste_threshold:
                recognized_gestures = ["Namaste"]
            elif dist < clap_threshold:
                recognized_gestures = ["Clapping"]
        
        # Display recognized gestures at top-left corner
        y_offset = 50
        for g in recognized_gestures:
            cv2.putText(img, f"{g}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 35
        
        # (Optional) To display AI narrative, uncomment the following lines:
        # if recognized_gestures:
        #     narrative = generate_narrative(recognized_gestures[0])
        #     detector.draw_multiline_text(img, narrative, (50, y_offset),
        #                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Hand Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()