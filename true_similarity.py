import numpy as np
import cv2

# Load images in grayscale
truth_img = cv2.imread('truth_path', cv2.IMREAD_GRAYSCALE)
pred_img = cv2.imread('pred_path', cv2.IMREAD_GRAYSCALE)

# Set threshold value
threshold_value = 254

# Convert grayscale images to binary images
ret, truth_img = cv2.threshold(truth_img, threshold_value, 255, cv2.THRESH_BINARY)
ret, pred_img = cv2.threshold(pred_img, threshold_value, 255, cv2.THRESH_BINARY)

# True positives: pixels that are present in both masks
tp = np.sum(np.logical_and(truth_img, pred_img))

# False positives: pixels that are present in pred_img but not in truth_img
fp = np.sum(np.logical_and(pred_img, np.logical_not(truth_img)))

# False negatives: pixels that are present in truth_img but not in pred_img
fn = np.sum(np.logical_and(truth_img, np.logical_not(pred_img)))

# True negatives: pixels that are not present in either mask
tn = np.sum(np.logical_and(np.logical_not(truth_img), np.logical_not(pred_img)))

# Calculate precision, recall and accuracy
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Show results
print('Precision: ', precision, '\nRecall: ', recall, '\nAccuracy: ', accuracy)