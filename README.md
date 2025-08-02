# Applied Machine Learning Projects

This repository contains a collection of machine learning projects implementing various algorithms and techniques for clustering, classification, and recommendation systems.

## Projects Overview

### 1. Data Reduction (`DataReduction.py`)
**Bisecting K-Means Clustering Algorithm**

A custom implementation of the bisecting K-means clustering algorithm that iteratively divides clusters to achieve the desired number of clusters.

**Features:**
- Variance threshold feature selection for dimensionality reduction
- Data normalization for improved clustering performance
- Sum of Squared Errors (SSE) calculation to determine optimal cluster splits
- Iterative cluster bisection until reaching target K clusters
- Cluster assignment and results export

**Use Case:** Unsupervised learning for data segmentation and pattern discovery in numerical datasets.

---

### 2. Drug Classification (`DrugClassification.py`)
**Text Classification using Decision Trees**

A text classification system that categorizes drugs based on their textual descriptions using traditional machine learning approaches.

**Features:**
- TF-IDF vectorization for text feature extraction
- Tweet tokenization for robust text preprocessing
- Variance threshold feature selection
- Decision Tree classifier with configurable parameters
- Binary classification output (0/1)

**Use Case:** Pharmaceutical data analysis and drug categorization based on textual descriptions.

---

### 3. Drug Classification - Neural Network (`DrugClassification-MLP.py`)
**Advanced Text Classification using Multi-Layer Perceptron**

An enhanced version of the drug classification system using neural networks for more sophisticated pattern recognition.

**Features:**
- Multi-Layer Perceptron (MLP) neural network classifier
- Advanced TF-IDF vectorization with optimized parameters
- Higher variance threshold for neural network optimization
- Prediction confidence scoring
- Comprehensive model performance reporting
- Support for multiple solvers (lbfgs, sgd, adam)

**Use Case:** Complex pharmaceutical text classification requiring deep pattern recognition capabilities.

---

### 4. Image Reduction (`ImageReduction.py`)
**Specialized Clustering for High-Dimensional Image Data**

A specialized bisecting K-means implementation optimized for image data analysis with advanced dimensionality reduction techniques.

**Features:**
- CSV format image data loading
- t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensionality reduction
- Furthest points initialization strategy for better cluster separation
- Variance threshold feature selection for high-dimensional data
- Data normalization and preprocessing pipeline
- Optimized for image feature clustering (default K=10)

**Use Case:** Computer vision applications, image segmentation, and high-dimensional image feature analysis.

---

### 5. Movie Recommender (`MovieRecommender.py`)
**Neural Network-Based Collaborative Filtering System**

A comprehensive movie recommendation system that combines content-based and collaborative filtering approaches using neural networks.

**Features:**
- Multi-source data integration (actors, directors, genres, tags)
- TF-IDF vectorization for movie content profiles
- User-movie rating matrix construction
- MLP Regressor for rating prediction
- Tag weighting system for enhanced recommendations
- Rating normalization and clamping (1.0-5.0 scale)

**Use Case:** Personalized movie recommendation systems and rating prediction for streaming platforms.

## Technical Stack

- **Python 3.x**
- **scikit-learn** - Machine learning algorithms and preprocessing
- **NLTK** - Natural language processing and tokenization
- **NumPy** - Numerical computing and array operations

## Key Algorithms Implemented

1. **Bisecting K-Means Clustering** - Custom implementation with SSE optimization
2. **Decision Tree Classification** - Text-based classification with TF-IDF
3. **Multi-Layer Perceptron** - Neural network for text and rating prediction
4. **t-SNE Dimensionality Reduction** - Non-linear dimensionality reduction for images
5. **Collaborative Filtering** - User-item matrix factorization for recommendations

---

*This repository demonstrates practical applications of machine learning algorithms across different domains including clustering, classification, and recommendation systems. Data not included*