import importlib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from imutils import paths
import seaborn as sns
import random
import time
from datetime import datetime
import requests
import zipfile
import json
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import io
from tqdm import tqdm


# State the path to the training set JSON file and load it to a variable
training_set_path = "mnist_handwritten_train.json"
with open(training_set_path, "r") as ts:
    training_set = json.load(ts)
 
# Take values and sort into acording list   
training_images = [np.array(i["image"],np.uint8).reshape((28,28)).flatten() for i in tqdm(training_set,desc="Loading Training Images...")]
training_labels = [x["label"] for x in tqdm(training_set,desc="Loading Training Labels...") ]

# Convert to nummpy arrays
training_images = np.array(training_images).astype('float32')
training_labels = np.array(training_labels)

#idfk
training_labels = training_labels.astype(int)
training_labels.reshape((training_labels.size,1))

print(f"Number of Images: {len(training_images)}")
print(f"Number of Labels: {len(training_labels)}")
#----------------------------TEST LISTS------------------------#

test_set_path = "mnist_handwritten_test.json"
with open(test_set_path, "r") as ts:
    test_set = json.load(ts)
    
# Take values and sort into acording list 

raw_images = [np.array(i["image"],np.uint8).reshape((28,28)) for i in test_set]
raw_labels = [[x["label"] for x in test_set ]]

raw_test_images = [np.array(i["image"],np.uint8).reshape((28,28)).flatten() for i in tqdm(test_set,desc="Loading Test Images...")]
raw_test_labels = [x["label"] for x in tqdm(test_set,desc="Loading Test Labels...") ]

# Convert to nummpy arrays
test_images = np.array(raw_test_images).astype('float32')
test_labels = np.array(raw_test_labels)

#idfk
test_labels = test_labels.astype(int)
test_labels.reshape((test_labels.size,1))


# Print number of values
print(f"Number of Images: {len(test_images)}")
print(f"Number of Labels: {len(test_labels)}")

# Actually train
t_start = datetime.now()
knn_model = cv2.ml.KNearest.create()

knn_model.train(training_images,cv2.ml.ROW_SAMPLE,training_labels)

# Number of Neighbors to consider
k_values = [x for x in range(1,11)]
k_result = []




# Accuracy Testing ->

ret,result,neighbours,dist = knn_model.findNearest(test_images[:100],k=245) 

result =  result.flatten()
for i in range(100):
    plt.figure(figsize=(6, 6))
    plt.imshow(raw_images[i])
    plt.title(f"Label: {raw_labels[0][i]} Model Prediction: {result[i]}")
    plt.axis("off")
    plt.show()
    
    
#-----------------------------------------------------------------------------------#

# for k in tqdm(k_values,desc="Testing..."):
#     ret,result,neighbours,dist = knn_model.findNearest(test_images,k=k)
#     k_result.append(result)
    
# flattened = []
# for res in k_result:
#     flat_result = [item for sublist in res for item in sublist]
#     flattened.append(flat_result)

# # Record end time and print how long training + prediction took
# end_datetime = datetime.now()
# print('Training Duration: ' + str(end_datetime - t_start))



# # Create empty lists to store accuracy results and confusion matrices for each K
# accuracy_res = []
# con_matrix = []

# # Loop over the results for each value of K
# for k_res in k_result:
#     # Define the class labels (e.g., 0 = Cat, 1 = Dog)
#     label_names = [0,1,2,3,4,5,6,7,8,9]

#     # Compute the confusion matrix for predictions vs. true labels
#     cmx = confusion_matrix(test_labels, k_res, labels=label_names)
#     con_matrix.append(cmx)

#     # Check which predictions match the true labels
#     matches = k_res == test_labels

#     # Count how many predictions were correct
#     correct = np.count_nonzero(matches)

#     # Calculate accuracy as a percentage
#     accuracy = correct * 100.0 / result.size
#     accuracy_res.append(accuracy)

# # Store the accuracy for each value of K in a dictionary (key = K, value = accuracy)
# res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}

# # Sort the results by K value to make them easier to read or plot
# list_res = sorted(res_accuracy.items())

# print("\nAccuracy per k:")
# for k, acc in list_res:
#     print(f"k = {k}: {acc:.2f}%")




