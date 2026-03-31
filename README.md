# ☕ Starbucks Reviews: NLP Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-150458?style=flat&logo=python&logoColor=white)](https://www.nltk.org/)

## 📖 Project Overview
This project performs an end-to-end Natural Language Processing (NLP) and Sentiment Analysis on a dataset of Starbucks customer reviews scraped from ConsumerAffairs. Following the **CRISP-DM** methodology, this analysis compares traditional Lexicon-based approaches (VADER, TextBlob) against classical Machine Learning algorithms (Naïve Bayes, Support Vector Machines) to classify customer sentiment and extract actionable business intelligence.

**Dataset:** [Starbucks Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset) (850 reviews)

## 🎯 Business Objectives
Starbucks' success relies heavily on the "Starbucks Experience." This project bridges the gap between unstructured text data and operational strategy by:
1. **Quantifying the Customer Experience (CX) Gap:** Translating raw text into Positive, Neutral, and Negative sentiments to evaluate brand health.
2. **Identifying Operational Hotspots:** Extracting N-grams (bigrams/trigrams) to pinpoint exact service bottlenecks (e.g., "drive thru", "get order") versus loyalty drivers.
3. **Benchmarking NLP Models:** Determining the most scalable and accurate methodology for real-time sentiment monitoring in a highly skewed real-world dataset.

## 📊 Key Insights & Business Intelligence
Through extensive Exploratory Data Analysis (EDA) and TF-IDF feature extraction, several critical business realities were uncovered:
* **Severe Negative Skew:** **77.95%** of verified reviews were Negative (1-2 stars), compared to just **17.35%** Positive and **4.69%** Neutral. This indicates the ConsumerAffairs platform is heavily utilized for grievance reporting.
* **Geographical Consistency:** Negative sentiment exceeded **70%** across all top 10 states analyzed, indicating a systemic rather than localized issue.
* **"Customer Service" is the Ultimate Differentiator:** The bigram `"customer service"` was the highest-weighted feature in *both* Negative (1.84 TF-IDF weight) and Positive (3.85 TF-IDF weight) reviews. Product quality ("great coffee") matters, but partner/employee interaction dictates the final sentiment.
* **Operational Bottlenecks:** Top negative phrases explicitly identified friction points: `"drive thru"`, `"get order"`, and `"time go"`. 

## ⚙️ Methodology

### 1. Data Pre-Processing
* **Dual-Dataset Strategy:** Created a training set of verified ratings for ML models, and a full set for unsupervised Lexicon analysis to prevent undersampling.
* **NLP Pipeline:** Lowercasing, URL/HTML/Number removal, punctuation stripping, custom stopword removal (e.g., 'starbucks'), POS tagging, and WordNet Lemmatization.

### 2. Lexicon-Based Modeling (VADER & TextBlob)
* **Performance:** Both models struggled with the extreme dataset bias. VADER severely underestimated negativity (predicting only **23.18%** Negative vs. the actual 77.95%), skewing heavily Positive (39.06%). TextBlob skewed heavily Neutral (43.88%). 
* **Takeaway:** Lexicon models struggle to detect implicit negativity, sarcasm, or polite complaints hidden within low-star reviews.

### 3. Machine Learning Modeling (TF-IDF)
Vectorized text using unigrams and bigrams, resulting in a high-dimensional sparse matrix of **27,823 features**. Tested multiple algorithms:
* **Multinomial Naïve Bayes**
* **Linear SVC** (Support Vector Classifier)
* **RBF SVC** & **Polynomial SVC**

## 📈 Model Evaluation & Results

| Model | Accuracy | Weighted F1-Score | Pos. Class Recall | Neg. Class Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVC** | **80.14%** | **0.73** | **0.12** | **1.00** |
| Naïve Bayes | 78.01% | 0.68 | 0.00 | 1.00 |
| RBF SVC | 78.01% | 0.68 | 0.00 | 1.00 |
| Polynomial SVC | 78.01% | 0.68 | 0.00 | 1.00 |

**The Imbalance Trap:** While the ML models achieved ~78-80% accuracy, this was highly deceptive. Because the dataset was nearly 78% negative, the models acted as "Negative Sentiment Detectors" (achieving 1.00 Recall for the Negative class) but failed completely on the minority Positive/Neutral classes. 

**LinearSVC** emerged as the best performer, being the *only* ML model robust enough in the high-dimensional space to correctly identify *any* true positive reviews, alongside the highest overall accuracy (80.14%). 

## 🚀 Next Steps & Future Work
To make the ML models production-ready and capable of balanced sentiment classification, future iterations must address the severe class imbalance. Planned improvements include:
* Implementing **SMOTE** (Synthetic Minority Over-sampling Technique) or random undersampling.
* Applying class-weight adjustments during SVM training.
* Exploring advanced word embeddings (Word2Vec) or Transformer models (BERT) to capture semantic context better than TF-IDF.

## 🛠️ Technologies Used
* **Languages:** Python
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn, WordCloud
* **NLP Libraries:** NLTK, TextBlob, VADER Sentiment
* **Machine Learning:** Scikit-Learn (TF-IDF, Naïve Bayes, SVM)

## 💻 How to Run This Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/starbucks-sentiment-analysis.git](https://github.com/yourusername/starbucks-sentiment-analysis.git)
   cd starbucks-sentiment-analysis

2. **Install required dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk vaderSentiment textblob wordcloud joblib

3. **Download NLTK Data (Run in Python):**
   ```Python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')

4. **Run the Notebook:**
Open the Jupyter Notebook (starbucks_sentiment_analysis.ipynb) to view the step-by-step code execution, visualizations, and model evaluations.

## ✍️ Author
Femi James
Data & Business Analyst | Integrated AI Specialist
