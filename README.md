# 📌 Machine Learning Fundamentals 

A comprehensive machine learning project covering **supervised learning** (regression & classification), **unsupervised learning** (K-Means clustering), **reinforcement learning** (Q-Learning), **deep learning / computer vision** (CNNs, transfer learning, YOLO), and **generative AI** (diffusion models, LLMs, TTS, super-resolution). Built as part of a structured ML course across multiple sessions.

## 📂 Project Structure

```
Machine-Learning-Projects/
├── Datasets/                              # Training datasets
├── diagrams/                              # Architecture diagrams used in Part 6
├── machine-learning-part1.ipynb           # Session 1: ML fundamentals & core concepts
├── machine-learning-part2.ipynb           # Session 2: Regression, classification, regularization
├── machine-learning-part3.ipynb           # Session 3: Classification, ensembles & model evaluation
├── machine-learning-part4.ipynb           # Session 4: Neural networks & deep learning
├── machine-learning-part5.ipynb           # Session 5: Advanced CNNs & computer vision
├── machine-learning-part6.ipynb           # Session 6: Generative AI & large language models
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

### Part 3 Classification, Ensembles & Model Evaluation

- **Classification Fundamentals** — binary (Breast Cancer) vs multiclass (Wine, Iris, Digits) problem setup and baselines
- **Logistic Regression Deep Dive** — sigmoid/derivative/logit plots, loss intuition, decision threshold tuning (best F1 at threshold ≈ 0.36), coefficients as odds ratios, regularization paths, calibration curves, linear vs polynomial features on the moons dataset
- **Decision Trees** — Gini vs entropy criteria, cost-complexity pruning (`ccp_alpha`), depth tuning, 5-fold cross-validation
- **Support Vector Machines** — hinge loss & margin intuition, linear vs RBF/poly/sigmoid kernels, effect of `C` and `gamma`, why scaling matters (75.0% → 94.4% on Wine), GridSearchCV tuning, support-vector visualization
- **Ensemble Methods** — Random Forest (OOB score, `max_features`, feature importances), AdaBoost (estimator weights, margins, staged improvement), Gradient Boosting (learning curves, partial dependence plots), XGBoost (eval history, gain importances), plus 2D decision regions and probability calibration
- **Performance Evaluation** — ROC/AUC and precision-recall curves, confusion matrices, precision/recall trade-offs across models
- **Class Imbalance Handling** — 90/10 imbalanced dataset with class weights and SMOTE oversampling
- **Titanic Survival Case Study** — the same problem solved four ways (Logistic Regression, Decision Tree, SVM, Random Forest) with shared preprocessing and confusion matrices
- **German Credit Risk Activity** — head-to-head comparison of LR / DT / SVM / RF on OpenML `credit-g` (mixed numeric + categorical, 30% positive rate), ranked by accuracy, precision, recall, F1, ROC-AUC, and training time

### Part 4 Neural Networks & Deep Learning

- **Perceptron from Scratch** — NumPy implementation, convergence on a linearly separable problem (100% accuracy) and its failure on XOR (50%), demonstrating the limits of a linear decision boundary
- **Backpropagation from Scratch** — a fully manual `DetailedNN` class with Xavier/He initialization, forward/backward passes, gradient caching, and gradient-flow visualization
- **Activation Functions** — sigmoid, tanh, ReLU, Leaky ReLU, Swish, GELU with their derivatives, range/zero-centering/smoothness comparison, and the vanishing-gradient problem
- **Keras MLP** — Digits classification with dropout, `EarlyStopping` and `ReduceLROnPlateau` callbacks (97.5% test accuracy)
- **CNNs** — 3-block MNIST architecture with batch normalization and dropout, trained with data augmentation (98.2% test accuracy), plus feature-map inspection
- **RNNs** — synthetic time series (trend + seasonality + noise) forecast with SimpleRNN, LSTM, GRU, BiLSTM, and stacked LSTM
- **Advanced Optimization** — SGD, SGD+Momentum, RMSprop, Adam, and AdamW compared on California Housing regression
- **Regularization Techniques** — baseline vs L2, dropout, batch normalization, and combined, with an L2 strength sweep

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

### Part 6 Generative AI & Large Language Models

A concept-and-architecture session built around annotated diagrams and interactive widgets rather than model training.

- **Generative AI Fundamentals** — discriminative P(Y\|X) vs generative P(X) models, latent spaces, embeddings, and probability distributions, with an interactive 2D latent-space plot showing concept clustering and interpolation
- **Diffusion Models (Stable Diffusion)** — forward noising vs reverse denoising, latent diffusion at 64×64×4 instead of 512×512×3 (48× cheaper), and the three components: CLIP text encoder (77×768), U-Net denoiser with cross- and self-attention (860M–2.6B params), and the VAE encoder/decoder
- **Generation Parameters** — classifier-free guidance scale, step count vs quality trade-off, and PNDM / DDIM / DPM samplers, each with a side-by-side comparison figure
- **Transformer Architecture** — encoder-only (BERT), decoder-only (GPT, LLaMA), and encoder-decoder (T5), with a focus on decoder-only generation
- **Attention Mechanism** — Q/K/V as a database lookup, the scaled dot-product formula, multi-head attention across 8–32 heads, and visualized attention patterns
- **Tokenization & Embeddings** — BPE / SentencePiece subword splitting, ~50K vocabularies, token IDs → dense vectors, and sinusoidal positional encoding
- **Autoregressive Generation** — next-token prediction loop, causal masking, and stop conditions
- **Sampling Strategies** — greedy, temperature, top-k, top-p (nucleus), and repetition penalty, with an interactive widget showing how temperature reshapes the probability distribution
- **Text-to-Audio** — the TTS pipeline (text encoder → acoustic model → vocoder), plus Bark (multilingual, music/SFX, speaker cloning) vs Meta's MMS-TTS (1,100+ languages)
- **Image Super-Resolution** — bicubic interpolation vs neural SR, the RRDB architecture behind Real-ESRGAN (dense connections, local/global residuals, PixelShuffle upsampling, GAN perceptual loss), and PSNR / SSIM quality metrics
- **Cross-Model Synthesis** — the unifying threads across all four model families: attention, transformers, latent representations, sampling, and self-supervised large-scale training

## 📊 Datasets Used

| Dataset | Task | Type | Source |
|---------|------|------|--------|
| California Housing | House price prediction | Regression | scikit-learn |
| FuelConsumptionCo2.csv | CO2 emission prediction | Regression | IBM |
| 1000_Companies.csv | Company profit prediction | Regression | — |
| Wine (sklearn) | Wine type classification | Classification | scikit-learn |
| Iris | Flower classification | Classification | scikit-learn |
| Wine Quality (Red) | Quality rating prediction | Classification | UCI ML Repository |
| Breast Cancer | Malignant vs benign tumours | Binary Classification | scikit-learn |
| Digits | Handwritten digit recognition | Classification | scikit-learn |
| Titanic | Passenger survival prediction | Binary Classification | seaborn |
| German Credit (credit-g) | Credit risk (good vs bad) | Binary Classification | OpenML |
| MNIST | Handwritten digit recognition | Deep Learning | Keras datasets |
| Synthetic Time Series | Sequence forecasting | Deep Learning (RNN) | Generated |
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
| Logistic Regression | Breast Cancer | 94.7% accuracy |
| Logistic Regression | Digits | 96.0% accuracy |
| Decision Tree (entropy) | Wine | 97.2% accuracy |
| SVM (RBF, tuned) | Wine | 98.9% CV accuracy |
| AdaBoost / Gradient Boosting / XGBoost | Breast Cancer | 95.6% accuracy, AUC up to 0.994 |
| Decision Tree (depth=5) | Titanic | 81.6% accuracy (best of 4 models) |
| Logistic Regression | German Credit | F1 = 0.593, ROC-AUC = 0.804 (best F1) |
| Keras MLP (128-64-32 + dropout) | Digits | 97.5% test accuracy |
| CNN (3 blocks + BN + augmentation) | MNIST | 98.2% test accuracy |
| GRU | Synthetic time series | Test MAE = 1.58 (best of 5 RNNs) |
| Dropout-regularized MLP | California Housing | Test MAE = 0.309 |
| VGG-Style CNN (3.25M params) | CIFAR-10 subset | 61.7% test accuracy |
| ResNet-Style CNN (216K params) | CIFAR-10 subset | 43.3% test accuracy |
| MobileNetV2 Transfer Learning (frozen) | CIFAR-10 subset | 61.5% test accuracy |
| MobileNetV2 Fine-Tuned (last 20 layers) | CIFAR-10 subset | 63.0% test accuracy (+1.5%) |
| YOLOv8n | Sample street image | bus 86.8%, persons 86.4% / 83.3% / 81.1% |

> Part 5 CNNs were trained on a 10,000-image CIFAR-10 subset (2,000 test images) on CPU, so accuracies reflect the limited training budget rather than the architectures' ceilings. Transfer learning used only 1,000 training images and reached comparable accuracy in a fraction of the epochs.

> Part 6 trains no models — it covers generative architectures through diagrams and interactive widgets, so it contributes no rows to the tables above.

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

# Extra dependencies for Part 6 (interactive widgets)
pip install ipywidgets
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
- xgboost - gradient boosted trees
- imbalanced-learn - SMOTE oversampling for class imbalance
- TensorFlow / Keras - MLPs, CNNs, RNNs, transfer learning, data augmentation
- OpenCV, Pillow - image loading, resizing, video I/O
- ultralytics - YOLOv8 detection, segmentation, pose estimation
- ipywidgets - interactive demos in the generative AI notebook

