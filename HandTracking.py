import cv2
import mediapipe as mp
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv
import os
import time
import math

# Initialize Google AI
load_dotenv()
my_api_key = os.getenv("GOOGLE_API_KEY")
# print(my_api_key)
genai.configure(api_key=my_api_key)

# models = genai.list_models()
# for model in models:
#     print(model.name)

class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=self.modelComp,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def draw_multiline_text(img, text, position, font, scale, color, thickness, line_spacing=30):
        # x, y = position
        print(position)
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (x, y + i * line_spacing), font, scale, color, thickness)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmList
def distance(point1, point2):
    return math.sqrt((point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
def recognize_gesture(lmList):
    """Recognize simple hand gestures based on landmark positions."""
    if not lmList:
            return "No hand detected"

        # Get fingertip Y-axis positions
    thumb_tip = lmList[4][2]
    index_tip = lmList[8][2]
    middle_tip = lmList[12][2]
    ring_tip = lmList[16][2]
    pinky_tip = lmList[20][2]

    # Get PIP joint Y-axis positions (for index, middle, ring, and pinky)
    index_pip = lmList[6][2]
    middle_pip = lmList[10][2]
    ring_pip = lmList[14][2]
    pinky_pip = lmList[18][2]

    # Define a threshold for the OK sign (you may need to adjust this value)
    ok_threshold = 30

    # OK Sign: Thumb and index tips are close forming a circle; other fingers extended
    if distance(lmList[4], lmList[8]) < ok_threshold and \
    middle_tip < middle_pip and ring_tip < ring_pip and pinky_tip < pinky_pip:
        return "OK Sign"

    # Thumbs Up: Thumb is extended (tip above the index PIP) while other fingers are folded
    if thumb_tip < index_pip and \
    middle_tip > index_pip and ring_tip > index_pip and pinky_tip > index_pip:
        return "Thumbs Up"

    # Open Palm: All fingers extended (tips above their PIP joints)
    if index_tip < index_pip and middle_tip < middle_pip and \
    ring_tip < ring_pip and pinky_tip < pinky_pip:
        return "Open Palm"

    # Fist: All fingers folded (tips below their PIP joints)
    if index_tip > index_pip and middle_tip > middle_pip and \
    ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Fist"

    # Pointing: Only index finger extended (others folded)
    if index_tip < index_pip and \
    middle_tip > middle_pip and ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Pointing"

    # Victory/Peace: Index and middle fingers extended; ring and pinky folded
    if index_tip < index_pip and middle_tip < middle_pip and \
    ring_tip > ring_pip and pinky_tip > pinky_pip:
        return "Victory/Peace"

    # Rock Sign: Index and pinky extended; middle and ring folded
    if index_tip < index_pip and \
    middle_tip > middle_pip and ring_tip > ring_pip and \
    pinky_tip < pinky_pip:
        return "Rock Sign"

    # Shaka: Thumb and pinky extended; index, middle, and ring fingers folded.
    # For thumb, compare with landmark 3 to check if it's extended.
    if thumb_tip < lmList[3][2] and \
    index_tip > index_pip and middle_tip > middle_pip and ring_tip > ring_pip and \
    pinky_tip < pinky_pip:
        return "Shaka"

    return "Unknown Gesture"

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

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        gesture = recognize_gesture(lmList)
        description = generate_narrative(gesture)

        cv2.putText(img, f"Gesture: {gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, description, (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # detector.draw_multiline_text(img, description, [50, 150], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()