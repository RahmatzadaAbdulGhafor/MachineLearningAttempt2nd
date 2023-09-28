import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import pickle








def create_training_data(X,y):
    DATADIR = "C:/Users/Ghafo/Desktop/projects/MachineLearningAttempt2nd/images/train"
    CATEGORIES = ["normal", "pneumonia"]
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to the images
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

            # plt.imshow(img_array, cmap="gray")
            # plt.show()

    random.shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()





if __name__ == "__main__":
    # feature set
    X = []
    # labels
    y = []

    IMG_SIZE = 150
    training_data = []

    create_training_data(X,y)

    # pickle_in = open("X.pickle", "rb")
    # X = pickle.load(pickle_in)
    #
    # pickle_in = open("y.pickle", "rb")
    # y = pickle.load(pickle_in)
    #
    # print("X PICKLE ", X[1])
    #
    # print("y PICKLE ", y[2])

