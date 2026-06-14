import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

def load_images_for_knn(base_path, size=(64, 64)):
    """Loads all image files under base_path, resizes them, grayscales, and flattens them."""
    data = []
    labels = []
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")
        
    for class_name in os.listdir(base_path):
        class_dir = os.path.join(base_path, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, size)
                            flat = resized.flatten()
                            data.append(flat)
                            labels.append(class_name)
                    except Exception as e:
                        print(f"Error reading image {img_path}: {e}")
                        
    return np.array(data), np.array(labels)

def train_and_evaluate_knn(train_path, test_path, k=5):
    print("Loading datasets for KNN...")
    X_train, y_train = load_images_for_knn(train_path)
    X_test, y_test = load_images_for_knn(test_path)
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Get unique class names
    classes = sorted(list(np.unique(y_train)))
    n_classes = len(classes)
    print("Classes:", classes)
    
    # Initialize and train the model
    print(f"Training KNN model with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nKNN Accuracy: {accuracy:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred))
    
    # Save folder directory check
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs/images", exist_ok=True)
    
    # Save the model
    model_path = os.path.join("models", "knn_friends_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    print(f"KNN model saved successfully to: {model_path}")
    
    # 1. Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join("docs", "images", "knn_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to: {cm_path}")
    plt.close()
    
    # 2. Multi-class ROC Curve and AUC
    # Binarize labels for multi-class ROC
    y_train_bin = label_binarize(y_train, classes=classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Wrap model with OneVsRestClassifier for multi-class probabilities
    ovr_knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=k))
    ovr_knn.fit(X_train, y_train_bin)
    y_score = ovr_knn.predict_proba(X_test)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", n_classes)
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"ROC curve of {classes[i]} (AUC = {roc_auc:.2f})")
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curves (KNN)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join("docs", "images", "knn_roc_curves.png")
    plt.savefig(roc_path)
    print(f"ROC Curves saved to: {roc_path}")
    plt.close()
    
    return accuracy

if __name__ == "__main__":
    train_dir = os.path.join("FriendsDataSet", "train")
    test_dir = os.path.join("FriendsDataSet", "test")
    train_and_evaluate_knn(train_dir, test_dir)
