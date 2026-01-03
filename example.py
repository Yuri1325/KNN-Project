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
import tqdm

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/RdukW75jUsonAnS20t3n_g/training-an-image-classifier-w-2025-05-22-t-10-27-47-719-z.zip"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open the zip file from the downloaded content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("cats_dogs")  # Extract to a target folder
    print("Download and extraction complete.")
else:
    print("Failed to download file:", response.status_code)

##########################################################################

# Define the path to the annotations JSON file
annotations_path = "cats_dogs/training-an-image-classifier-w-2025-05-22-t-10-27-47-719-z/_annotations.json"

# Load the JSON file
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Now safely access the first five entries
first_five = {k: annotations["annotations"][k] for k in list(annotations["annotations"])[:5]}
first_five

#-------------------------------------------------------------------#

base_folder = "cats_dogs/training-an-image-classifier-w-2025-05-22-t-10-27-47-719-z"

# Path to the annotations JSON file
annotations_path = os.path.join(base_folder, "_annotations.json")

# Load the JSON data
with open(annotations_path, "r") as f:
    annotations = json.load(f)

print("Annotations loaded successfully!")

#-------------------------------------------------------------------#

# Pick a random image from the annotations
random_image_name = random.choice(list(annotations["annotations"].keys()))

# Get the label for that image
label = annotations["annotations"][random_image_name][0]["label"]

print(f"Random image selected: {random_image_name}")
print(f"Label: {label}")

#---------------------------------------------------------------------#

# Construct full path to image
image_path = os.path.join(base_folder, random_image_name)

# Read image using OpenCV
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
print("Full image path:", image_path)

#--------------------------------------------------------------------#

# Convert image color from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot using matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

# if you plot img(BGR image), you will observe a difference in the color space 
plt.imshow(img)
plt.axis('off')
plt.title(f"Label: {label}")
plt.show()      
#------------------------------------------------------------------------#

sample_image = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.imshow(sample_image, cmap = "gray")
plt.show()

#-------------------------------------------------------------------#

sample_image = cv2.resize(img_rgb, (32, 32))
plt.imshow(sample_image, cmap = "gray")
plt.show()

#-----------------------------------------------------------------------#

pixels = sample_image.flatten()
pixels

#----------------------------------#

# Get all image file paths from the dataset folder
image_paths = list(paths.list_images(base_folder))

# Create empty lists to store image data and corresponding labels
train_images = []
train_labels = []

# Extract the list of class labels (e.g., ['dog', 'cat']) from the annotations
class_object = annotations['labels']

#------------------------------------------------------------------------------#

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from tqdm import tqdm
# Process each image with a progress bar
for image_path in tqdm(image_paths, desc="Loading images"):
    filename = os.path.basename(image_path)

    # Skip if not in annotation
    if filename not in annotations["annotations"]:
        continue  # silently skip

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    pixels = image.flatten()

    # Get label
    tmp_label = annotations["annotations"][filename][0]['label']
    label = class_object.index(tmp_label)

    # Append
    train_images.append(pixels)
    train_labels.append(label)

train_images = np.array(train_images).astype('float32')
train_labels = np.array(train_labels)

train_labels = train_labels.astype(int)
train_labels = train_labels.reshape((train_labels.size,1))
print("First 5 labels:\n", train_labels[:5])

print(f"Number of images: {len(train_images)}")
print(f"Number of labels: {len(train_labels)}")

test_size = 0.2
train_samples, test_samples, train_labels, test_labels = train_test_split(
    train_images, train_labels, test_size=test_size, stratify=train_labels,random_state=0)

# Record the start time to measure training duration
start_datetime = datetime.now()

# Create a KNN model using OpenCV's machine learning module
knn = cv2.ml.KNearest_create()

# Train the model using training samples and corresponding labels
# cv2.ml.ROW_SAMPLE specifies that each row in train_samples is a separate sample
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

# Define different values of K to evaluate
k_values = [1, 2, 3, 4, 5]
k_result = []  # To store the prediction results for each value of K

# Loop through each K value and test the model on test samples
for k in k_values:
    ret, result, neighbours, dist = knn.findNearest(test_samples, k=k)
    k_result.append(result)  # Save the result for this value of K

# Flatten the result arrays for easier comparison later
flattened = []
for res in k_result:
    # Each `res` is a 2D array; flatten it into a 1D list
    flat_result = [item for sublist in res for item in sublist]
    flattened.append(flat_result)

# Record end time and print how long training + prediction took
end_datetime = datetime.now()
print('Training Duration: ' + str(end_datetime - start_datetime))

# Create empty lists to store accuracy results and confusion matrices for each K
accuracy_res = []
con_matrix = []

# Loop over the results for each value of K
for k_res in k_result:
    # Define the class labels (e.g., 0 = Cat, 1 = Dog)
    label_names = [0, 1]

    # Compute the confusion matrix for predictions vs. true labels
    cmx = confusion_matrix(test_labels, k_res, labels=label_names)
    con_matrix.append(cmx)

    # Check which predictions match the true labels
    matches = k_res == test_labels

    # Count how many predictions were correct
    correct = np.count_nonzero(matches)

    # Calculate accuracy as a percentage
    accuracy = correct * 100.0 / result.size
    accuracy_res.append(accuracy)

# Store the accuracy for each value of K in a dictionary (key = K, value = accuracy)
res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}

# Sort the results by K value to make them easier to read or plot
list_res = sorted(res_accuracy.items())

print("\nAccuracy per k:")
for k, acc in list_res:
    print(f"k = {k}: {acc:.2f}%")