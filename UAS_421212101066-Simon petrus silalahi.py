import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.decomposition import PCA

# Load dataset
digits = datasets.load_digits()
data = digits.data
target = digits.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Reshape data
X_train = X_train.reshape(-1, 8, 8)
X_test = X_test.reshape(-1, 8, 8)

# Calculate HOG features
def extract_features(imgs):
    features = []
    for img in imgs:
        # Reshape to 8x8
        img = img.reshape((8, 8))
        
        # Compute HOG features
        hog_features = hog.compute(img)
        features.append(hog_features)
        
    return np.array(features)

# Initiate HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Extract HOG features from training and testing sets
X_train_hog = extract_features(X_train)
X_test_hog = extract_features(X_test)

# Create a Support Vector Machine Classifier
clf = svm.SVC(gamma=0.001)

# Train the model using the training sets
clf.fit(X_train_hog, y_train)

# Make predictions on the testing set
pred = clf.predict(X_test_hog)

# Print classification report
print(classification_report(y_test, pred))

# Print confusion matrix
print(confusion_matrix(y_test, pred))

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')