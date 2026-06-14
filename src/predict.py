import os
import random
import argparse
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing import load_image, to_grayscale

# Class names mapping from indices to human-readable names
CLASS_MAP = {
    0: "Chandler Bing",
    1: "Joey Tribbiani",
    2: "Monica Geller",
    3: "Phoebe Buffay",
    4: "Rachel Green",
    5: "Ross Geller"
}

CLASS_KEYS = [
    'chandler-bing',
    'joey-tribbiani',
    'monica-geller',
    'phoebe-buffay',
    'rachel-green',
    'ross-geller'
]

def load_models():
    """Loads the serialized KNN and CNN models from the models/ directory."""
    cnn_path = os.path.join("models", "cnn_friends_model.keras")
    knn_path = os.path.join("models", "knn_friends_model.pkl")
    
    cnn_model = None
    knn_model = None
    
    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}...")
        cnn_model = load_model(cnn_path)
    else:
        print(f"Warning: CNN model file not found at {cnn_path}. You should train it first using train_cnn.py")
        
    if os.path.exists(knn_path):
        print(f"Loading KNN model from {knn_path}...")
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)
    else:
        print(f"Warning: KNN model file not found at {knn_path}. You should train it first using train_knn.py")
        
    return cnn_model, knn_model

def get_random_scene_image(random_scene_dir):
    """Finds a random image recursively inside the directory."""
    images = []
    for root, _, files in os.walk(random_scene_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    if not images:
        raise FileNotFoundError(f"No images found in random scene folder: {random_scene_dir}")
    return random.choice(images)

def predict_single_image(image_path, cnn_model, knn_model):
    """Runs predictions on a single image using both KNN and CNN models."""
    # Load original RGB image
    img = load_image(image_path)
    
    # 1. Preprocess for CNN
    cnn_pred_label = "Model not loaded"
    cnn_confidence = 0.0
    if cnn_model is not None:
        cnn_img = cv2.resize(img, (224, 224))
        cnn_img = cnn_img / 255.0
        cnn_input = np.expand_dims(cnn_img, axis=0)
        
        cnn_preds = cnn_model.predict(cnn_input, verbose=0)
        cnn_class_idx = np.argmax(cnn_preds, axis=1)[0]
        cnn_pred_label = CLASS_MAP.get(cnn_class_idx, "Unknown")
        cnn_confidence = cnn_preds[0][cnn_class_idx]
        
    # 2. Preprocess for KNN
    knn_pred_label = "Model not loaded"
    if knn_model is not None:
        gray_img = to_grayscale(img)
        knn_img = cv2.resize(gray_img, (64, 64))
        knn_input = knn_img.flatten().reshape(1, -1)
        
        # Predict class key
        knn_class_key = knn_model.predict(knn_input)[0]
        # Map class key back to human-readable format
        if knn_class_key in CLASS_KEYS:
            idx = CLASS_KEYS.index(knn_class_key)
            knn_pred_label = CLASS_MAP.get(idx, knn_class_key)
        else:
            knn_pred_label = knn_class_key
            
    # Visualize and save results
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # Extract actual label from file path if it matches folder structure
    actual_label = "Unknown"
    parent_dir = os.path.basename(os.path.dirname(image_path))
    if parent_dir in CLASS_KEYS:
        actual_label = CLASS_MAP.get(CLASS_KEYS.index(parent_dir))
    
    title_str = (
        f"Actual: {actual_label}\n"
        f"CNN Predict: {cnn_pred_label} ({cnn_confidence:.2%})\n"
        f"KNN Predict: {knn_pred_label}"
    )
    
    plt.title(title_str, fontsize=14, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    
    # Save the prediction visualization
    os.makedirs("docs/images", exist_ok=True)
    pred_save_path = os.path.join("docs", "images", "last_prediction.png")
    plt.savefig(pred_save_path)
    print(f"Prediction result visualization saved to: {pred_save_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Friends Character Classifier Inference CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to a custom image file.")
    args = parser.parse_args()
    
    # Load trained models
    cnn_m, knn_m = load_models()
    
    if cnn_m is None and knn_m is None:
        print("Error: Neither CNN nor KNN model is loaded. Please train the models first.")
        exit(1)
        
    # Pick target image
    target_image = args.image
    if target_image is None:
        random_dir = os.path.join("FriendsDataSet", "random-scene")
        if not os.path.exists(random_dir):
            random_dir = os.path.join("FriendsDataSet", "test")
            
        print(f"No custom image provided. Selecting random image from {random_dir}...")
        try:
            target_image = get_random_scene_image(random_dir)
            print(f"Selected image: {target_image}")
        except FileNotFoundError as e:
            print(str(e))
            print("Please provide a valid image using --image <path> or extract dataset.")
            exit(1)
            
    # Run prediction
    predict_single_image(target_image, cnn_m, knn_m)
