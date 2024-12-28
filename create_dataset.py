import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'

# Initialize data and labels
data = []
labels = []

# Iterate over each class directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory: {dir_path}")
        continue

    # Iterate over images in the directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            print(f"Failed to read image: {img_full_path}")
            continue

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting image to RGB: {img_full_path}, {e}")
            continue

        # Process the image with MediaPipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                # Normalize landmarks relative to the bounding box
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(dir_)
        else:
            print(f"No hand landmarks detected in: {img_full_path}")

# Save processed data
output_path = 'data.pickle'
with open(output_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data saved to {output_path}")
