import cv2
import matplotlib.pyplot as plt


# Extract the keypoints of an image using SIFT algorithm
def extract_sift_features(img):
    sift_initialize = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift_initialize.detectAndCompute(img, None)
    return key_points, descriptors


# Plot the image and the keypoints
def plot_sift_features(img1, img2, key_points):
    return plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))


# Read the images
Image1 = cv2.imread('bigben1.jpg')
Image2 = cv2.imread('bigben2.jpg')

# Convert the images to grayscale
Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

# Extract the key points and descriptors of each image
Image1_key_points, Image1_descriptors = \
    extract_sift_features(Image1_gray)
Image2_key_points, Image2_descriptors = \
    extract_sift_features(Image2_gray)

# Show the key points of each image
plot_sift_features(Image1_gray, Image1, Image1_key_points)
plt.show()
plot_sift_features(Image2_gray, Image2, Image2_key_points)
plt.show()

# Find the distance between the key points using the Manhattan distance
distance = cv2.NORM_L2
# Find the match between descriptors of two points
bfmatcher = cv2.BFMatcher(distance)
matches = bfmatcher.match(Image1_descriptors, Image2_descriptors)
# The matches sorted based on the Manhattan distance
matches = sorted(matches, key=lambda match: match.distance)
# Connect the top 100 matched points
matched_img = cv2.drawMatches(Image1, Image1_key_points, Image2, Image2_key_points,
                              matches[:100], Image2.copy())
# Plot the image
plt.figure(figsize=(100, 300))
plt.show()
plt.imshow(matched_img)
plt.show()
