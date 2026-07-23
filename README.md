# 🐱🐶 Cats vs Dogs Classification using SVM

A Machine Learning project that classifies images of **cats** and **dogs** using **Support Vector Machine (SVM)** with **HOG feature extraction**, **PCA**, and **GridSearchCV** for hyperparameter tuning.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Model | SVM (RBF Kernel) |
| Feature Extraction | HOG |
| Dimensionality Reduction | PCA |
| Best Parameters | C = 10, Gamma = 0.001 |
| Test Accuracy | **76.62%** |
| Training Images | 4,000 (2,000 Cats + 2,000 Dogs) |

---

## 📂 Dataset

- **Dataset:** Microsoft Research / Kaggle Cats vs Dogs
- **Total Images:** 25,000
- **Classes:** Cat (0), Dog (1)

> The dataset is not included due to its large size.

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── svm_cats_dogs.py
└── data/
    ├── Cat/
    └── Dog/
```

---

## ⚙️ Installation

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python svm_cats_dogs.py \
--data_dir data/PetImages \
--sample_size 4000 \
--n_components 200
```

---

## 🔄 Workflow

- Load images
- Extract HOG features
- Scale features
- Apply PCA
- Train SVM (RBF Kernel)
- Tune using GridSearchCV
- Evaluate and save the model

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Joblib

---

## 📌 Note

Classical ML models like SVM typically achieve **65–80% accuracy** on this dataset. This project achieved **76.62%** test accuracy.

---

## 📜 License

This project is for educational purposes. The dataset belongs to Microsoft Research/Kaggle.
