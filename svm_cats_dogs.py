"""
SVM Classifier for Cats vs Dogs (Kaggle "Dogs vs. Cats" dataset)
==================================================================

Dataset structure expected (after unzipping Kaggle's train.zip):

    train/
        cat.0.jpg
        cat.1.jpg
        ...
        dog.0.jpg
        dog.1.jpg
        ...

Pipeline:
    1. Load images, resize to a fixed size, convert to grayscale
    2. Extract HOG (Histogram of Oriented Gradients) features
       -> HOG works far better than raw pixels for SVMs because it
          captures edge/shape structure and is much lower-dimensional.
    3. Scale features (SVMs are sensitive to feature scale)
    4. Reduce dimensionality with PCA (speeds up training, reduces overfitting)
    5. Train an SVM (RBF kernel) with GridSearchCV to tune C and gamma
    6. Evaluate on a held-out test set (accuracy, confusion matrix, report)
    7. Save the trained model + scaler + PCA to disk for later use

Usage:
    python svm_cats_dogs.py --data_dir /path/to/train --sample_size 4000
"""

import os
import argparse
import time
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

IMG_SIZE = 64  # images are resized to IMG_SIZE x IMG_SIZE before feature extraction


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_image_paths(data_dir, sample_size=None):
    """Collect balanced lists of cat/dog file paths.

    Supports two layouts:
      1. Kaggle style: data_dir contains files named cat.N.jpg / dog.N.jpg
      2. Microsoft style: data_dir contains subfolders Cat/ and Dog/
         (e.g. PetImages/Cat/*.jpg, PetImages/Dog/*.jpg)
    """
    cat_dir = os.path.join(data_dir, "Cat")
    dog_dir = os.path.join(data_dir, "Dog")

    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        # Microsoft-style folder layout
        cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)]
        dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)]
    else:
        # Kaggle-style flat layout with cat./dog. filename prefixes
        all_files = os.listdir(data_dir)
        cat_files = [os.path.join(data_dir, f) for f in all_files if f.lower().startswith("cat")]
        dog_files = [os.path.join(data_dir, f) for f in all_files if f.lower().startswith("dog")]

    if sample_size is not None:
        per_class = sample_size // 2
        cat_files = cat_files[:per_class]
        dog_files = dog_files[:per_class]

    paths = cat_files + dog_files
    labels = [0] * len(cat_files) + [1] * len(dog_files)  # 0 = cat, 1 = dog
    return paths, labels


# ----------------------------------------------------------------------
# 2. Feature extraction (HOG)
# ----------------------------------------------------------------------
def extract_hog_features(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    )
    return features


def build_feature_matrix(paths, labels, img_size=IMG_SIZE):
    X, y = [], []
    for i, (p, label) in enumerate(zip(paths, labels)):
        feats = extract_hog_features(p, img_size)
        if feats is None:
            continue  # skip unreadable/corrupt images
        X.append(feats)
        y.append(label)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(paths)} images")
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# 3. Training pipeline
# ----------------------------------------------------------------------
def train(data_dir, sample_size=4000, n_components=200, model_out="svm_cats_dogs.joblib"):
    print("Loading image paths...")
    paths, labels = load_image_paths(data_dir, sample_size)
    print(f"Total images: {len(paths)} (cats: {labels.count(0)}, dogs: {labels.count(1)})")

    print("Extracting HOG features (this can take a few minutes)...")
    t0 = time.time()
    X, y = build_feature_matrix(paths, labels)
    print(f"Feature extraction done in {time.time() - t0:.1f}s. Feature vector length: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Reducing dimensionality with PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Explained variance retained: {pca.explained_variance_ratio_.sum():.3f}")

    print("Running grid search for best SVM hyperparameters...")
    param_grid = {
        "C": [1, 10, 50],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf"],
    }
    grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train_pca, y_train)

    print(f"Best params: {grid.best_params_}")
    best_model = grid.best_estimator_

    print("Evaluating on test set...")
    y_pred = best_model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Saving model bundle to {model_out}")
    joblib.dump(
        {"model": best_model, "scaler": scaler, "pca": pca, "img_size": IMG_SIZE},
        model_out,
    )
    return best_model, scaler, pca


# ----------------------------------------------------------------------
# 4. Inference on a single new image
# ----------------------------------------------------------------------
def predict_image(image_path, model_path="svm_cats_dogs.joblib"):
    bundle = joblib.load(model_path)
    model, scaler, pca, img_size = (
        bundle["model"],
        bundle["scaler"],
        bundle["pca"],
        bundle["img_size"],
    )
    feats = extract_hog_features(image_path, img_size)
    if feats is None:
        raise ValueError(f"Could not read image: {image_path}")
    feats_scaled = scaler.transform([feats])
    feats_pca = pca.transform(feats_scaled)
    pred = model.predict(feats_pca)[0]
    label = "dog" if pred == 1 else "cat"
    print(f"{image_path} -> predicted: {label}")
    return label


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SVM to classify cats vs dogs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Kaggle train/ folder")
    parser.add_argument("--sample_size", type=int, default=4000, help="Total images to use (balanced)")
    parser.add_argument("--n_components", type=int, default=200, help="PCA components")
    parser.add_argument("--model_out", type=str, default="svm_cats_dogs.joblib")
    args = parser.parse_args()

    train(args.data_dir, args.sample_size, args.n_components, args.model_out)
