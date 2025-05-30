# Data-Mining
This repository contains my solutions to assignments from Data Mining course, implementing various machine learning algorithms and text classification techniques.

## 📋 Assignments

### HW0: Data Processing & TFIDF Implementation
- **Task**: Implement TFIDF calculation from scratch for news article classification
- **Dataset**: 1000 news articles across 5 categories (sport, business, politics, entertainment, tech)
- **Implementation**: 
  - Text preprocessing (lowercase, punctuation removal, tokenization, stemming)
  - Manual TFIDF matrix computation (1000x1000)
  - Top 3 frequent words and highest TFIDF words per category
- **Deliverables**: Jupyter notebook, TFIDF matrix file, frequency/scores JSON files

### HW1: Tree-Based Classification Models
- **Task**: Implement and evaluate decision trees, random forests, and boosting models
- **Dataset**: News classification (1000 training articles, 681 test articles)
- **Models**: Decision Trees, Random Forest, AdaBoost, GradientBoosting
- **Evaluation**: 5-fold cross-validation with parameter tuning (criterion, min_samples_leaf, max_features, n_estimators)
- **Deliverables**: Code notebook, results description, predicted labels CSV

### HW2: Neural Networks & Feature Engineering
- **Task**: Build neural networks with advanced feature engineering techniques
- **Dataset**: News classification with extended feature exploration
- **Implementation**:
  - 2-hidden layer neural networks (128 neurons each)
  - Feature methods: CountVectorizer, TFIDF, GloVe embeddings, BERT
  - Hyperparameter tuning: learning rates, optimizers (SGD, Adam, RMSprop)
- **Evaluation**: 5-fold cross-validation with accuracy comparison
- **Deliverables**: Code notebook, performance analysis, predicted labels CSV

### Final Project: Crop Yield & Price Prediction
- **Task**: Time series forecasting for agricultural data using multiple approaches
- **Dataset**: Historical crop yield and price data with weather conditions
- **Models**: ARIMA, LSTM, Prophet, XGBoost, Random Forest, Ensemble methods
- **Objective**: Predict crop yields and prices to help farmers make informed decisions
- **Evaluation**: RMSE, MAE, R² metrics with comprehensive model comparison
- **Impact**: Address market volatility and reduce price manipulation by intermediaries

## 🛠 Technologies Used
- **Languages**: Python
- **Libraries**: Scikit-learn, TensorFlow/PyTorch, Pandas, NumPy, NLTK, Prophet, XGBoost
- **Techniques**: Text processing, Feature engineering, Cross-validation, Time series analysis

## 📊 Key Results
- **HW0**: Successfully implemented TFIDF from scratch with proper text preprocessing
- **HW1**: Random Forest and boosting methods achieved best performance for tree-based models
- **HW2**: BERT embeddings significantly outperformed traditional TFIDF features
- **Final Project**: XGBoost and Random Forest achieved superior results (R² > 0.93) compared to time series models