import os
import cv2
import numpy as np

def color_histogram_of_test_image(test_src_image):
    image = test_src_image

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chans = cv2.split(hsv_image)
    colors = ('h', 's', 'v')  # hue, saturation, value

    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist.flatten())

    # Normalize the histogram features
    features = np.array(features)
    features /= np.sum(features)

    # Save normalized histogram features to a file
    with open('test.data', 'w') as myfile:
        myfile.write(','.join(map(str, features)))

def color_histogram_of_training_image(img_name):
    # Determine data source from image file name
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'
    else:
        data_source = 'unknown'

    try:
        # Load image
        image = cv2.imread(img_name)

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        chans = cv2.split(hsv_image)
        colors = ('h', 's', 'v')  # hue, saturation, value

        features = []
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist.flatten())

        # Normalize the histogram features
        features = np.array(features)
        features /= np.sum(features)

        # Save normalized histogram features and label to a file
        with open('training.data', 'a') as myfile:
            feature_data = ','.join(map(str, features))
            myfile.write(f'{feature_data},{data_source}\n')

    except Exception as e:
        print(f"Error processing {img_name}: {str(e)}")

def training():
    # List of color directories
    color_directories = ['red', 'yellow', 'green', 'orange', 'white', 'black', 'blue']

    # Process each color directory
    for color_dir in color_directories:
        dir_path = f'./training_dataset/{color_dir}/'
        if not os.path.exists(dir_path):
            continue

        # Iterate through images in the directory
        for f in os.listdir(dir_path):
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                img_path = os.path.join(dir_path, f)
                color_histogram_of_training_image(img_path)

if __name__ == '__main__':
    training()
