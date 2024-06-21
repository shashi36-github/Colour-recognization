import cv2
import os
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import sys

def load_source_image():
    if len(sys.argv) > 1:
        try:
            source_image = cv2.imread(sys.argv[1])
        except Exception as e:
            print(f"Error loading image from argument: {e}")
            source_image = cv2.imread('black_cat.jpg')  # Load default image on error
    else:
        source_image = cv2.imread('black_cat.jpg')  # Load default image if no argument provided
    return source_image

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
    source_image = load_source_image()
    prediction = 'n.a.'

    check_training_data()

    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main('training.data', 'test.data')
    print('Detected color is:', prediction)

    cv2.putText(source_image, 'Prediction: ' + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, (200, 200, 200), 2)

    # Display the resulting frame
    cv2.imshow('Color Classifier', source_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
