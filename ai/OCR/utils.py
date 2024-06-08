import numpy as np
import cv2
from matplotlib import pyplot as plt


# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis("off")

    # Display the image.
    ax.imshow(im_data, cmap="gray")

    plt.show()


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return newImage
import numpy as np


def preProcessing(img):
    """A function used to pre-process images to make it suitable to work with.

    Parameters: img (matrix): matrix image
    Returns: imgThres(matrix): Processed matrix image
    """
    # 1 - Convert to grey level
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2 - Denoise image
    imgBlur = cv2.fastNlMeansDenoising(imgGray, h=10)
    # imgBlur2 = cv2.GaussianBlur(imgGray,(5,5),1)  # another way of denoising

    # 3 - Removing unnecessary details such as writing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(imgBlur, cv2.MORPH_CLOSE, kernel, iterations=5)

    # 4 - Edge detection using canny
    imgCanny = cv2.Canny(morph, 75, 200)
    # imgCanny2 = cv2.Canny(imgBlur,75,200) # without morph

    # 5- Extra preprocessing Dilation and Erosion( Mostly we don't needs them )
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres


def camScanner(img_path, width=720, height=1280):

    img = cv2.imread(img_path)

    # Process image before detecting
    processedImg = preProcessing(img)

    # Get corners points
    corners = cornerDetector(processedImg)

    # Wrap the image into flat x, y directions
    imgWarped = warper(img, corners, width, height)

    return imgWarped


def cornerDetector(img):
    """A function used to highlight all the contours in the image and
    find the corners of the biggest one.

    Parameters: img (matrix): matrix image
    Returns: corners(Array): return the biggest contours corner pixel points
    """
    # biggest contours corner points saved in an array
    corners = np.array([])
    # initalize the max area
    maxArea = 0
    # calling method find contours to find all contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # loop through each contour in the image
    for cnt in contours:
        # svae its area size
        area = cv2.contourArea(cnt)
        # if area smaller than 6k ignore it
        if area > 6000:
            # if we want to see all contours uncomment this
            # cv2.drawContours(imgCopy, cnt, -1, (0, 0, 255), 2)
            # Calculate the contour curves length we need it for approximation points (the bigger the accurate)
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            # An approximation of the edges points ( corner points)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # if this contour has bigger area and it has 4 points edges ( meaning it is a square such as paper)
            if area > maxArea and len(approx) == 4:
                corners = approx
                maxArea = area
    return corners


def sortCorners(corners):
    """A function used to reorder the corners point to
    match the exact order required for the wrapper function.

    Parameters: corners (array): corners of the paper
    Returns: sortedCorners(array): returns sortedd array of corners pixel points
    """
    # Just reshaping before manuplations
    corners = corners.reshape((4, 2))
    # create empty new corners
    sortedCorners = np.zeros((4, 1, 2), np.int32)
    # sum each corner width + height => smallest is at the top left[0,0]
    # biggest will be bottom right [width, height]
    # we want to make the corner like this order [[0, 0], [width, 0], [0, height], [width, height]]
    add = corners.sum(1)
    # smallest
    sortedCorners[0] = corners[np.argmin(add)]
    # biggest
    sortedCorners[3] = corners[np.argmax(add)]
    # The difference between width - height for the remaining points:
    #  if the result (height - width) is negative(minimum) => will be at [width, 0]
    #  if the result (height - width) is positive(maximum) => will be at [0, height]
    diff = np.diff(corners, axis=1)
    sortedCorners[1] = corners[np.argmin(diff)]
    sortedCorners[2] = corners[np.argmax(diff)]

    return sortedCorners


def warper(img, corners, width, height):
    """A function used to transform the document into fully flat in x and y directions.

    Parameter: img (matrix): matrix image
    Parameter: corners (array): corners of the paper

    Returns: corners(array): fully flat in x and y image
    """
    # Calling sort corner to satisfy wrapper function
    corners = sortCorners(corners)
    # set the points
    points1 = np.float32(corners)
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Wrapper transform computer transform matrix
    matrix = cv2.getPerspectiveTransform(points1, points2)
    # result
    result = cv2.warpPerspective(img, matrix, (width, height))

    # if we want to better cut the noise edges
    # imgCropped = result[10:result.shape[0]-10, 10:result.shape[1]-10]
    # imgCropped = cv2.resize(imgCropped,(width,height))

    return result
