import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import requests
import time
from time import sleep

class GestureRecognizer:
    def _init_(self):
        # Initialize MediaPipe Hand Detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize gesture database
        self.gesture_database = {}
        self.load_gestures()
        
        # Training mode variables
        self.is_training = False
        self.current_gesture_name = ""
        self.training_frames = []
        
        # Default ESP8266 configuration
        self.esp8266_ip = "172.20.10.3"  # Default ESP8266's IP address
        self.last_gesture = None
        self.gesture_cooldown = 1.0  # Cooldown in seconds
        self.last_gesture_time = 0
        
    def load_gestures(self):
        """Load saved gestures from file if exists"""
        if os.path.exists('gestures.pkl'):
            with open('gestures.pkl', 'rb') as f:
                self.gesture_database = pickle.load(f)
                print(f"Loaded {len(self.gesture_database)} gestures")
    
    def save_gestures(self):
        """Save gestures to file"""
        with open('gestures.pkl', 'wb') as f:
            pickle.dump(self.gesture_database, f)
            print("Gestures saved successfully")
    
    def get_hand_features(self, hand_landmarks):
        """Extract features from hand landmarks"""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def calculate_gesture_similarity(self, features1, features2):
        """Calculate similarity between two gesture features"""
        return np.linalg.norm(np.array(features1) - np.array(features2))
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize gesture based on saved database"""
        if not self.gesture_database:
            return "No gestures trained"
        
        current_features = self.get_hand_features(hand_landmarks)
        
        best_match = None
        min_distance = float('inf')
        
        for gesture_name, gesture_features_list in self.gesture_database.items():
            for gesture_features in gesture_features_list:
                distance = self.calculate_gesture_similarity(current_features, gesture_features)
                if distance < min_distance:
                    min_distance = distance
                    best_match = gesture_name
        
        return best_match if min_distance < 2.0 else "Unknown"
    
    def start_training(self, gesture_name):
        """Start training mode for a new gesture"""
        self.is_training = True
        self.current_gesture_name = gesture_name
        self.training_frames = []
        print(f"Training mode started for gesture: {gesture_name}")
    
    def add_training_frame(self, hand_landmarks):
        """Add a training frame for the current gesture"""
        if self.is_training:
            features = self.get_hand_features(hand_landmarks)
            self.training_frames.append(features)
            print(f"Captured frame {len(self.training_frames)}/30")
    
    def finish_training(self):
        """Save the trained gesture"""
        if self.is_training and len(self.training_frames) > 0:
            self.gesture_database[self.current_gesture_name] = self.training_frames
            self.save_gestures()
            print(f"Gesture '{self.current_gesture_name}' trained successfully")
        self.is_training = False
        self.training_frames = []

    def delete_gesture(self, gesture_name):
        """Delete a gesture from the database"""
        if gesture_name == "delete all":
            self.gesture_database.clear()
            self.save_gestures()
            print("All gestures deleted successfully")
        elif gesture_name in self.gesture_database:
            del self.gesture_database[gesture_name]
            self.save_gestures()
            print(f"Gesture '{gesture_name}' deleted successfully")
        else:
            print(f"Gesture '{gesture_name}' not found")
            
    def send_gesture_to_esp8266(self, gesture):
        """Send gesture command to ESP8266"""
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
            
        if gesture != self.last_gesture and gesture != "Unknown":
            try:
                response = requests.post(
                    f"http://{self.esp8266_ip}/gesture",
                    data=gesture,
                    timeout=1
                )
                if response.status_code == 200:
                    print(f"Gesture {gesture} sent successfully")
                else:
                    print(f"Failed to send gesture: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with ESP8266: {e}")
            
            self.last_gesture = gesture
            self.last_gesture_time = current_time

def main():
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    print("\nControls:")
    print("ESC - Exit")
    print("T - Enter training mode")
    print("F - Capture frame (manual frame capture for training)")
    print("S - Save current training frames")
    print("D - Delete a gesture")
    print("I - Set ESP8266 IP address")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.hands.process(rgb_frame)
        
        # Display training mode status
        if recognizer.is_training:
            cv2.putText(frame, f"Training: {recognizer.current_gesture_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(recognizer.training_frames)}/30", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                recognizer.mp_draw.draw_landmarks(
                    frame, hand_landmarks, recognizer.mp_hands.HAND_CONNECTIONS)
                key = cv2.waitKey(1) & 0xFF
                if recognizer.is_training:
                    if len(recognizer.training_frames) < 30:
                        if key == ord('f'):
                            recognizer.add_training_frame(hand_landmarks)
                    else:
                        recognizer.finish_training()
                else:
                    gesture = recognizer.recognize_gesture(hand_landmarks)
                    cv2.putText(frame, f"Gesture: {gesture}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    recognizer.send_gesture_to_esp8266(gesture)
        
        cv2.imshow('Gesture Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('t'):
            gesture_name = input("Enter name for the new gesture: ")
            recognizer.start_training(gesture_name)
        elif key == ord('f'):
            print("Frame captured")
        elif key == ord('s'):
            recognizer.finish_training()
        elif key == ord('d'):
            gesture_name = input("Enter the name of the gesture to delete ('delete all' to remove all): ")
            recognizer.delete_gesture(gesture_name)
        elif key == ord('i'):
            new_ip = input("Enter ESP8266 IP address: ")
            recognizer.esp8266_ip = new_ip
            print(f"ESP8266 IP set to: {new_ip}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
