import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Loads an image from the specified path and converts it to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(img):
    """Converts an RGB image to Grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def blur_image(gray_img, kernel_size=(5, 5)):
    """Applies Gaussian Blur to a grayscale image to reduce noise."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)

def equalize_histogram(gray_img):
    """Applies Histogram Equalization to enhance contrast."""
    return cv2.equalizeHist(gray_img)

def apply_sobel(gray_img):
    """Applies Sobel filter to detect horizontal, vertical, and combined edges."""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    
    sobel_combined = cv2.bitwise_or(np.uint8(abs_sobel_x), np.uint8(abs_sobel_y))
    return np.uint8(abs_sobel_x), np.uint8(abs_sobel_y), sobel_combined

def apply_canny(gray_img, low_threshold=50, high_threshold=150):
    """Applies Canny edge detection after smoothing."""
    blurred = blur_image(gray_img)
    return cv2.Canny(blurred, low_threshold, high_threshold)

def extract_contours(img):
    """Extracts external contours from the image and draws them on a copy of the original."""
    gray = to_grayscale(img)
    edges = apply_canny(gray)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contoured_img = img.copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)
    return contoured_img, len(contours)

def split_channels(img):
    """Splits the image into R, G, B channels."""
    # OpenCV splits as B, G, R, but our input is RGB
    r, g, b = cv2.split(img)
    return r, g, b

def visualize_preprocessing_steps(image_path):
    """Visualizes the core preprocessing steps side-by-side."""
    img = load_image(image_path)
    img_resized = cv2.resize(img, (224, 224))
    
    gray = to_grayscale(img_resized)
    blurred = blur_image(gray)
    equalized = equalize_histogram(gray)
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_resized)
    plt.title("Original (Resized)")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(blurred, cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(equalized, cmap="gray")
    plt.title("Histogram Equalization")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def visualize_all_filters(image_path):
    """Visualizes advanced filter outputs side-by-side."""
    img = load_image(image_path)
    img_resized = cv2.resize(img, (224, 224))
    
    gray = to_grayscale(img_resized)
    _, _, sobel_combined = apply_sobel(gray)
    canny = apply_canny(gray)
    contoured_img, num_contours = extract_contours(img_resized)
    r, g, b = split_channels(img_resized)
    
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(img_resized)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(2, 4, 2)
    plt.imshow(sobel_combined, cmap="gray")
    plt.title("Sobel Edges")
    plt.axis("off")
    
    plt.subplot(2, 4, 3)
    plt.imshow(canny, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")
    
    plt.subplot(2, 4, 4)
    plt.imshow(contoured_img)
    plt.title(f"Contours (N={num_contours})")
    plt.axis("off")
    
    plt.subplot(2, 4, 5)
    plt.imshow(r, cmap="Reds")
    plt.title("Red Channel")
    plt.axis("off")
    
    plt.subplot(2, 4, 6)
    plt.imshow(g, cmap="Greens")
    plt.title("Green Channel")
    plt.axis("off")
    
    plt.subplot(2, 4, 7)
    plt.imshow(b, cmap="Blues")
    plt.title("Blue Channel")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
