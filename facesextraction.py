import cv2
import dlib
import os

# input/output folder path
path = "test/wade"
outPath = "test_faces/wade"
# iterate over all the images in the folder
for image_path in os.listdir(path):

    input_path = os.path.join(path, image_path)
    # Load CNN face detector
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    # Load image
    img = cv2.imread(input_path)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find faces in image
    rects = dnnFaceDetector(gray, 1)
    left, top, right, bottom = 0, 0, 0, 0

    # For each face 'rect' provides face location in image as pixel loaction
    for (i, rect) in enumerate(rects):
        left = rect.rect.left()  # x1
        top = rect.rect.top()  # y1
        right = rect.rect.right()  # x2
        bottom = rect.rect.bottom()  # y2
        width = right - left
        height = bottom - top

        # Crop image
        img_crop = img[top:top + height, left:left + width]
        fullpath = os.path.join(outPath, 'cropped_' + image_path)
        # save crop image with person name as image name
        cv2.imwrite(fullpath, img_crop)