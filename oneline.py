import numpy as np
import cv2

# Load the image
image_path = 'eifellog.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
#blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(image, 100, 250)

# Apply Morphological thinning
def thinning(img):
    # Create a skeletonized version of the image
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

# Apply the thinning function to the edge-detected image
thin_edges = thinning(edges)

# Save or display the result
cv2.imwrite('thin_edges_image.png', thin_edges)

# If you want to display the result using OpenCV (optional):
# cv2.imshow('Thinned Edges', thin_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
