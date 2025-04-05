from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the pre-trained model
classifier = load_model('D:\MobileNet\Emotion_Detection.h5')

# Load the pre-trained face cascade classifier
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Define class labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open the webcam
cap = cv2.VideoCapture(1)

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 3)
            
            # Append the predicted label to the list of predicted labels
            predicted_labels.append(label)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 3)

    cv2.imshow('Emotion Detector', frame)
    
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Generate random true labels for demonstration
    true_labels = np.random.choice(class_labels, size=len(predicted_labels))
    
    # Check if enough predictions have been made for calculation
    if len(true_labels) > 0:
        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', labels=np.unique(predicted_labels))
        recall = recall_score(true_labels, predicted_labels, average='weighted', labels=np.unique(true_labels))

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
