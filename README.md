# 📌 Machine Learning Fundamentals 

A comprehensive machine learning project covering **supervised learning** (regression & classification), **unsupervised learning** (K-Means clustering), **reinforcement learning** (Q-Learning), and **deep learning / computer vision** (CNNs, transfer learning, YOLO). Built as part of a structured ML course across multiple sessions.

## 📂 Project Structure

```
Machine-Learning-Projects/
├── Datasets/                              # Training datasets
├── machine-learning-part1.ipynb           # Session 1: ML fundamentals & core concepts
├── machine-learning-part2.ipynb           # Session 2: Regression, classification, regularization
├── machine-learning-part3.ipynb           # Session 3: Applied classification case studies
├── machine-learning-part4.ipynb           # Session 4: Neural networks & deep learning intro
├── machine-learning-part5.ipynb           # Session 5: Advanced CNNs & computer vision
├── requirements.txt                       # Python dependencies
└── .gitignore
```

## 📌 What's Covered

### Part 1 ML Fundamentals & Core Concepts

- **Linear Regression** on California Housing dataset (EDA, correlation heatmap, train/test split, evaluation metrics, residual analysis, learning curves)
- **Unsupervised Learning**: K-Means customer segmentation on synthetic data (elbow method, cluster visualization)
- **Reinforcement Learning**: Q-Learning grid world navigation (reward curves, policy visualization, state visits, animated trajectories)
- **Features & Labels** with Wine classification using Random Forest (100% test accuracy)
- **Complete ML Pipeline**: Data loading → EDA → splitting → scaling → training → evaluation
- **Underfitting vs Overfitting**: Polynomial degree comparison (degree 1, 5, 15) on synthetic sine data
- **Data Handling**: Loading, EDA, and preprocessing on synthetic customer data (missing value imputation, outlier removal via IQR, feature scaling, categorical encoding)

### Part 2 Regression, Classification & Regularization

- **Simple Linear Regression** — CO2 emissions from engine size (R² = 0.76)
- **Multivariable Linear Regression** — CO2 from 6 vehicle features (R² = 0.90)
- **Linear Regression Assumptions** — visual examples of linearity, homoscedasticity, normality (good vs violated)
- **Comprehensive CO2 Analysis** — EDA, VIF multicollinearity check, OLS vs Ridge vs Lasso vs ElasticNet comparison, feature importance, residual diagnostics
- **Polynomial Regression** — capturing non-linear relationships, degree comparison (underfitting → good fit → overfitting)
- **Evaluation Metrics Deep Dive** — MSE, RMSE, MAE, R², Adjusted R² across normal, outlier, and heteroscedastic datasets
- **Company Profit Prediction** — linear vs polynomial regression on 1000 Companies dataset (R² = 0.98)
- **Logistic Regression** — sigmoid function, binary classification on Iris, multi-class decision boundaries
- **Decision Trees** — Iris classification with decision boundary visualization
- **Wine Quality Classification** — multi-class on UCI Red Wine dataset with detailed classification reports
- **Bias-Variance Tradeoff** — training on 50 random datasets to decompose bias², variance, and irreducible error
- **Regularization** — Ridge (L2), Lasso (L1), ElasticNet with alpha tuning via GridSearchCV
- **Model Saving/Loading** with `joblib`

### Part 5 Advanced CNNs & Computer Vision

- **CNN Architecture Evolution** — hand-built VGG-style stack (3 conv blocks, 3.25M params) vs ResNet-style network with residual skip connections and batch norm (216K params), trained head-to-head on a CIFAR-10 subset for 15 epochs
- **Transfer Learning** — MobileNetV2 pre-trained on ImageNet as a frozen feature extractor with a custom classification head (CIFAR-10 images upscaled to 96×96)
- **Fine-Tuning** — unfreezing the last 20 base layers and retraining at a reduced learning rate (1e-5) for a further accuracy gain
- **Image Data Augmentation** — `ImageDataGenerator` with rotation, shift, zoom, and horizontal flip; side-by-side training of the same CNN with and without augmentation to show the regularization effect on validation loss
- **Pre-trained ImageNet Classification** — ResNet50 with `decode_predictions` for top-5 labelling of arbitrary images
- **Feature Map Visualization** — extracting and plotting intermediate conv-layer activations to see what the network learns
- **Object Detection with YOLOv8** — `ultralytics` YOLOv8n for image detection (bounding boxes + confidences), video-file detection with annotated output, real-time webcam detection, and the dataset/label/`data.yaml` layout for custom training
- **Instance Segmentation** — YOLOv8n-seg for per-object masks
- **Pose Estimation** — YOLOv8n-pose for 17 body keypoints

## 📊 Datasets Used

| Dataset | Task | Type | Source |
|---------|------|------|--------|
| California Housing | House price prediction | Regression | scikit-learn |
| FuelConsumptionCo2.csv | CO2 emission prediction | Regression | IBM |
| 1000_Companies.csv | Company profit prediction | Regression | — |
| Wine (sklearn) | Wine type classification | Classification | scikit-learn |
| Iris | Flower classification | Classification | scikit-learn |
| Wine Quality (Red) | Quality rating prediction | Classification | UCI ML Repository |
| Synthetic Customer Data | Customer segmentation | Clustering | Generated |
| GridWorld | Navigation agent | Reinforcement Learning | Generated |
| CIFAR-10 | Image classification (10 classes) | Deep Learning / CV | Keras datasets |
| ImageNet (weights only) | Pre-trained backbones (MobileNetV2, ResNet50) | Transfer Learning | Keras Applications |
| COCO (weights only) | Detection / segmentation / pose | Object Detection | Ultralytics YOLOv8 |

## 🔑 Key Results

| Model | Dataset | Metric |
|-------|---------|--------|
| Linear Regression (1 feature) | CO2 Emissions | R² = 0.762 |
| Multivariable Linear Regression | CO2 Emissions | R² = 0.903 |
| OLS / Ridge / Lasso / ElasticNet | CO2 Emissions | R² ≈ 0.903 |
| Linear Regression | California Housing | R² = 0.576 |
| Linear Regression | Company Profit | R² = 0.983 |
| Random Forest Classifier | Wine (sklearn) | 100% accuracy |
| Logistic Regression | Iris (multi-class) | 97.3% accuracy |
| Decision Tree (depth=2) | Iris | 96.0% accuracy |
| Q-Learning Agent | 5×5 GridWorld | Avg reward = 92.94 |
| VGG-Style CNN (3.25M params) | CIFAR-10 subset | 61.7% test accuracy |
| ResNet-Style CNN (216K params) | CIFAR-10 subset | 43.3% test accuracy |
| MobileNetV2 Transfer Learning (frozen) | CIFAR-10 subset | 61.5% test accuracy |
| MobileNetV2 Fine-Tuned (last 20 layers) | CIFAR-10 subset | 63.0% test accuracy (+1.5%) |
| YOLOv8n | Sample street image | bus 86.8%, persons 86.4% / 83.3% / 81.1% |

> Part 5 CNNs were trained on a 10,000-image CIFAR-10 subset (2,000 test images) on CPU, so accuracies reflect the limited training budget rather than the architectures' ceilings. Transfer learning used only 1,000 training images and reached comparable accuracy in a fraction of the epochs.

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/Sanduni-Gamage/ML-Journey.git
cd Machine-Learning-Projects

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Extra dependencies for Part 5 (computer vision)
pip install opencv-python ultralytics
```

YOLOv8 weights (`yolov8n.pt`, `yolov8n-seg.pt`, `yolov8n-pose.pt`) download automatically on first use.

## 🛠 Tech Stack

- **Python 3.8+**
- pandas, numpy - data manipulation
- matplotlib, seaborn - visualization
- scikit-learn - ML models, preprocessing, and evaluation
- statsmodels - VIF and statistical tests
- scipy - residual analysis and statistical distributions
- joblib - model persistence
- TensorFlow / Keras - CNNs, transfer learning, data augmentation
- OpenCV, Pillow - image loading, resizing, video I/O
- ultralytics - YOLOv8 detection, segmentation, pose estimation

