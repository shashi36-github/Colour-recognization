import cv2
import os
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

def check_training_data():
    PATH = './training.data'
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print('Training data is ready, classifier is loading...')
    else:
        print('Training data is being created...')
        with open('training.data', 'w'):
            pass  # Create an empty file
        color_histogram_feature_extraction.training()
        print('Training data is ready, classifier is loading...')

def main():
    cap = cv2.VideoCapture(0)  # Change to 1 if your webcam is on index 1
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    check_training_data()
    prediction = 'n.a.'

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.putText(frame, 'Prediction: ' + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, (200, 200, 200), 2)

        # Display the resulting frame
        cv2.imshow('Color Classifier', frame)

        color_histogram_feature_extraction.color_histogram_of_test_image(frame)

        prediction = knn_classifier.main('training.data', 'test.data')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
