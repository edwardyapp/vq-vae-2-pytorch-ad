# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

from skimage.metrics import structural_similarity
import cv2
import numpy as np

# Load images
before = cv2.imread('/home/sfmt/PycharmProjects/vq-vae-2-pytorch/reconstructed_00000_00000.png')
after = cv2.imread('/home/sfmt/PycharmProjects/vq-vae-2-pytorch/restored_00000_00000.png')

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
# (score, diff) = structural_similarity(before_gray, after_gray, full=True,
#                                       win_size=11,
#             gaussian_weights=True,
#             multichannel=True,
#             sigma=1.5)

diff = np.abs(before_gray - after_gray)

# score = np.sum(diff, axis=0).flatten()

# print("Image Similarity: {:.4f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

height, width, channels = before.shape
# biggestArea = 0.001 # percentage

for c in contours:
    area = cv2.contourArea(c)
    if area > 40000:
    # if area < biggestArea / 100 * height * width:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff', diff)
cv2.imshow('diff_box', diff_box)
cv2.imshow('mask', mask)
cv2.imshow('filled after', filled_after)
cv2.waitKey()

# Calculate the per-element absolute difference between
# two arrays or between an array and a scalar
diff = 255 - cv2.absdiff(before, after)



# mask = np.zeros(before.shape, dtype='uint8')

cv2.imshow('diff', diff)
cv2.waitKey()