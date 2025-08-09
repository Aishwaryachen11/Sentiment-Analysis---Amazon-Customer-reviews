# 📊 Sentiment Analysis on Amazon Fine Food Reviews

## 📌 Overview
This project performs **sentiment analysis** on the Amazon Fine Food Reviews dataset to classify reviews into **Positive**, **Negative**, or **Neutral** categories.  
It includes **data preprocessing, feature extraction, model training, evaluation, explainability, and live prediction demos**.

## 🧠 Key Features
- **Dataset:** Amazon Fine Food Reviews (`Reviews.csv`)
- **Label mapping:**
  - 1–2 stars → Negative
  - 3 stars → Neutral
  - 4–5 stars → Positive
- **Preprocessing:** Lowercasing, stopword removal, optional lemmatization
- **Feature extraction:** TF-IDF (unigrams + bigrams, 40k max features)
- **Models:** Logistic Regression (One-vs-Rest), Multinomial Naive Bayes
- **Class imbalance handling:** `class_weight='balanced'` and optional undersampling
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Explainability:** SHAP bar plot of top impactful words
- **Inference:** Predict sentiment & probability for new reviews

## 🧹 Preprocessing
Before training our sentiment analysis models, we performed a structured text-cleaning and dataset-preparation process to ensure consistent, high-quality inputs and balanced class representation.  

### Steps Taken
1. **Lowercasing text**  
   All review text was converted to lowercase to remove case sensitivity issues (e.g., “Good” and “good” are treated the same).  
2. **Removing punctuation & numbers**  
   Non-alphabetic characters, punctuation marks, and numerical digits were stripped out to keep only meaningful textual content.  
3. **Tokenization**  
   Sentences were split into individual tokens (words) to enable further processing at the word level.  
4. **Stopword removal**  
   Common but uninformative words (e.g., *and*, *the*, *is*) were removed to reduce noise and improve feature quality.  
5. **Lemmatization**  
   Each token was converted to its base or dictionary form (e.g., *running* → *run*) to normalize variations of the same word and reduce vocabulary size.  
6. **Initializing lemmatizer and stopwords**  
   The NLTK WordNet lemmatizer was initialized along with the English stopword set to perform the above steps consistently.  
7. **Handling class imbalance**  
   The dataset had a strong skew toward positive reviews. We used **RandomUnderSampler** to downsample the majority classes so the model sees equal representation from Negative, Neutral, and Positive classes.  
8. **Shuffling dataset**  
   After balancing, the dataset was shuffled to ensure the training process does not learn unintended order-based patterns.  

## Train/Test Split + TF-IDF Features
After preprocessing, we prepared the data for machine learning model training and evaluation.
### Steps Taken
1. **Train/Test split**  
   - The cleaned dataset was split into **80% training** and **20% testing** sets.  
   - We used **stratified sampling** to preserve the original class distribution across train and test sets.  
   - This ensures that each sentiment category (Negative, Neutral, Positive) is proportionally represented in both subsets.
2. **TF-IDF vectorization**  
   - We transformed the cleaned text into numeric feature vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.  
   - TF-IDF helps measure how important a word is to a document relative to the entire dataset, downweighting common words and upweighting unique ones.
3. **Feature engineering details**  
   - **N-grams:** Included both unigrams (single words) and bigrams (two-word sequences) to capture short context.  
   - **Maximum features:** Limited the vocabulary to the **40,000 most frequent tokens** to keep the feature space manageable and improve training speed.  
   - **Minimum document frequency:** Ignored words appearing in fewer than **3 documents** to filter out extremely rare terms.  
4. **Output shape**  
   - Final training feature matrix: `(454,756 rows × 40,000 features)`  
   - Final testing feature matrix: `(113,690 rows × 40,000 features)`
  
## Model Training & Evaluation
We trained two baseline machine learning models using the TF-IDF features: **Logistic Regression** and **Multinomial Naive Bayes**.

### 1. Logistic Regression (TF-IDF, class_weight=balanced)
- **Reasoning:** Logistic Regression is a strong baseline for text classification due to its ability to handle high-dimensional sparse data.
- **class_weight=balanced** was used to account for any residual class imbalance after preprocessing.
- **Performance on Test Set:**
    - **Accuracy:** 85.03%
    - **Precision (Negative / Neutral / Positive):** 0.69 / 0.38 / 0.97
    - **Recall (Negative / Neutral / Positive):** 0.82 / 0.62 / 0.88
    - **F1-score (Negative / Neutral / Positive):** 0.75 / 0.47 / 0.92
    - **Macro Avg F1:** 0.71

**Key Observations:**
- Very high performance for the **Positive** class.
- Neutral sentiment detection remains challenging (lower precision & recall).
- Balanced weight helps boost recall for underrepresented classes.
- 
### 2. Multinomial Naive Bayes (TF-IDF)
- **Reasoning:** Naive Bayes is a simple yet effective probabilistic classifier for text data.
- Works well with term frequency features like TF-IDF, especially for linearly separable text data.
- **Performance on Test Set:**
    - **Accuracy:** 84.90%
    - **Precision (Negative / Neutral / Positive):** 0.82 / 0.69 / 0.85
    - **Recall (Negative / Neutral / Positive):** 0.50 / 0.07 / 0.99
    - **F1-score (Negative / Neutral / Positive):** 0.62 / 0.13 / 0.92
    - **Macro Avg F1:** 0.56

**Key Observations:**
- Naive Bayes performs extremely well on **Positive** reviews but struggles heavily with **Neutral** classification (very low recall).
- Precision for Negative reviews is higher than Logistic Regression, but recall is worse.
- The assumption of feature independence may limit Naive Bayes in nuanced sentiment tasks.

### 💡 Summary
- **Best overall performer:** Logistic Regression — better balance between precision and recall across classes.
- **Naive Bayes** — faster and simpler, but weaker on Neutral sentiment.
- Both models achieve ~85% accuracy, but class-level metrics highlight where improvements are needed.

## Confusion Matrix & Feature Insights
### 1. Confusion Matrix
We generated a confusion matrix for both models to visualize misclassifications.
- **Logistic Regression**:
    - Correctly identifies the majority of Positive and Negative reviews.
    - Neutral reviews are frequently misclassified as either Positive or Negative, confirming the precision/recall challenges observed earlier.
- **Multinomial Naive Bayes**:
    - Overwhelming bias toward classifying reviews as Positive.
    - Very low recall for Neutral sentiment (most Neutral reviews classified as Positive).
### 2. Top Features from Logistic Regression
We extracted the top **positive** and **negative** words for each class based on model coefficients.  
This helps explain what textual patterns the model relies on for predictions.
#### Class: **Negative**
- **Top Positive (strong indicators of Negative sentiment):**
    ```
    might good, threw, return, bland, weak, stale, yuck, great review, never buy, tasteless, disgusting, unfortunately, horrible, disappointment, disappointed, awful, disappointing, two star, terrible, worst
    ```
- **Top Negative (words that *decrease* likelihood of being Negative):**
    ```
    great, delicious, perfect, best, love, highly recommend, excellent, hooked, wonderful, favorite, good, amazing, nice, awesome, skeptical, well worth, yummy, even better, downside, addicted
    ```
#### Class: **Neutral**
- **Top Positive (strong indicators of Neutral sentiment):**
    ```
    anything special, least favorite, guess, bag arrived, great either, hoping, mediocre, tea purchased, plus side, average, think buy, little disappointed, bad great, decent, nothing special, get wrong, unfortunately, however, okay, three star
    ```
- **Top Negative (words that *decrease* likelihood of being Neutral):**
    ```
    perfect, one star, four star, delicious, hooked, yum, well worth, excellent, highly recommended, wonderful, highly, love, skeptical, star instead, highly recommend, great, ever, can wait, definitely recommend, better expected
    ```
#### Class: **Positive**
- **Top Positive (strong indicators of Positive sentiment):**
    ```
    yummy, awesome, highly, even better, pleased, favorite, yum, amazing, well worth, four star, skeptical, wonderful, highly recommend, best, hooked, excellent, love, perfect, great, delicious
    ```
- **Top Negative (words that *decrease* likelihood of being Positive):**
    ```
    worst, three star, disappointing, disappointed, unfortunately, disappointment, terrible, awful, two star, horrible, okay, stale, never buy, weak, least favorite, disgusting, bland, mediocre, great review, hoping
    ```
**Key Takeaways:**
- Positive sentiment is often tied to strong, clear praise words (“perfect”, “delicious”, “love”).
- Negative sentiment is linked to explicit dissatisfaction (“worst”, “terrible”, “disappointing”).
- Neutral sentiment often comes from moderate language or mixed feedback (“average”, “okay”, “nothing special”).

## 🔍 SHAP Analysis – Explaining Logistic Regression Predictions
We used **SHAP (SHapley Additive exPlanations)** to interpret the Logistic Regression model trained on **TF-IDF features**.  
Instead of a full force plot or summary with all features, we generated a **bar plot** showing the average absolute SHAP value for the top words across the dataset.  
### Purpose
- Identify which words have the **largest impact** on the model’s predictions.
- Translate technical model outputs into **business-friendly insights**.
### Results – Bar Plot
![SHAP Bar Plot](SHAP-Barplot.png)

**Interpretation:**
- Words like *"great"*, *"delicious"*, and *"love"* have a strong **positive influence** toward predicting Positive sentiment.
- Words such as *"worst"*, *"disappointing"*, and *"terrible"* strongly push the prediction toward Negative sentiment.
- More neutral/mid-impact words like *"average"* or *"mediocre"* influence the Neutral sentiment classification.
**Key Benefit:**  
This interpretability step makes the model **transparent** for non-technical stakeholders, allowing them to see **why** a review was classified in a certain way.

## Predict Sentiment for New Reviews (with Confidence Scores)
After training and evaluating the models, we tested the **Logistic Regression (TF-IDF)** model on new, unseen review text to simulate real-world usage.  
For each review:
- The model predicts the **sentiment class** (`Positive`, `Negative`, `Neutral`).
- We also output the **confidence score** for the predicted class.
- Full **probability distribution** across all classes is shown for transparency.

### Example Predictions
**Review:** *"The product quality is amazing! Will definitely buy again."*  
**Predicted:** **Positive** (confidence: `0.974`)  
**Probabilities:** `{'Negative': '0.006', 'Neutral': '0.020', 'Positive': '0.974'}`  

**Review:** *"Worst purchase ever. Completely useless and a waste of money."*  
**Predicted:** **Negative** (confidence: `0.999`)  
**Probabilities:** `{'Negative': '0.999', 'Neutral': '0.001', 'Positive': '0.000'}`  

**Review:** *"It's okay, delivery was on time but packaging could be better."*  
**Predicted:** **Neutral** (confidence: `0.634`)  
**Probabilities:** `{'Negative': '0.341', 'Neutral': '0.634', 'Positive': '0.025'}`  

### Purpose
- Demonstrates how the model can be integrated into a **production application** for live sentiment classification.
- Confidence scores allow **threshold tuning** (e.g., flagging low-confidence predictions for human review).

