import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
import cv2

def LoG_filter(image, sigma, size=None, pre_blur=False):
    # Optionally apply Gaussian blur to the image to reduce noise
    if pre_blur:
        image = gaussian_filter(image, sigma=sigma/2)

    # Generate LoG kernel
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7

    if size % 2 == 0:
        size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel to ensure that the sum of absolute values is 1
    kernel /= np.sum(np.abs(kernel))

    # Perform convolution with the LoG kernel
    result = convolve(image, kernel)

    return result

# Example usage:
image = cv2.imread(r"eifel.png", cv2.IMREAD_GRAYSCALE)  # Replace with your image path
sigma = 3
filtered_image = LoG_filter(image, sigma)

# Save the filtered image in full resolution
cv2.imwrite('LoG_Filtered_Image.png', filtered_image)

# Optionally, you can still plot the original and filtered images for visual inspection
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=16)
plt.axis('off')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('LoG Filtered Image', fontsize=16)
plt.axis('off')

plt.show()