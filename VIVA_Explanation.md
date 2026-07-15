# 📚 Viva Explanation — Customer Churn Prediction ML Project

---

## ❓ What is Customer Churn Prediction?

**Churn** means when a customer **stops using a company's service** — they leave or cancel.

**Churn Prediction** is the task of **predicting in advance** which customers are likely to leave, so the company can take action to retain them (e.g., offer discounts or better plans).

**Example:** A telecom company like Jazz or Zong wants to know: *"Which customers will cancel their subscription next month?"*

**Why it matters:**
- Keeping an existing customer costs **5x less** than getting a new one
- Companies can offer targeted promotions to at-risk customers
- It helps increase revenue and customer satisfaction

---

## 🤔 Supervised vs Unsupervised Learning

| Feature | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Labels** | Has labeled data (Churn = Yes/No) | No labels at all |
| **Goal** | Predict output for new data | Find hidden patterns/groups |
| **Examples** | Logistic Regression, Decision Tree, KNN, MLP | K-Means, Hierarchical Clustering |
| **Evaluation** | Accuracy, Confusion Matrix | Inertia, Dendrogram |
| **Used when** | You have historical labeled data | You want to discover groups |

---

## 📌 Why Each Algorithm Is Used

---

### 1️⃣ Logistic Regression

- **What it does:** Predicts the **probability** of churn (0 or 1) using a sigmoid function
- **Why used:** Simple, fast, works well for binary classification problems like Churn (Yes/No)
- **Output:** Probability between 0 and 1; threshold at 0.5 → Churn or Not
- **Best for:** Linearly separable data

---

### 2️⃣ Linear Regression (used as classifier)

- **What it does:** Fits a line to data and predicts a continuous value
- **Why used here:** To show contrast with Logistic Regression — Linear Regression is NOT normally used for classification, but we threshold its output at 0.5
- **Limitation:** May predict values < 0 or > 1, making it less ideal than Logistic Regression for this task
- **Used for:** Understanding the difference between regression and classification

---

### 3️⃣ Decision Tree Classifier

- **What it does:** Splits data into branches using feature conditions (like if-else rules)
- **Why used:** Easy to visualize and interpret; no need for feature scaling
- **Example rule:** IF `monthly_charges > 70` AND `tenure < 12` → **Churn = Yes**
- **Best for:** Non-linear data; great for viva explanation

---

### 4️⃣ K-Nearest Neighbors (KNN)

- **What it does:** Classifies a new point based on the **majority vote** of its K nearest neighbors
- **Why used:** Simple, no training phase; good baseline model
- **Parameter K:** K=5 means look at 5 closest customers and take majority vote
- **Best for:** Small to medium datasets; requires feature scaling (which we applied)

---

### 5️⃣ MLP — Multi-Layer Perceptron (Neural Network)

- **What it does:** A neural network with hidden layers that learns non-linear patterns
- **Architecture used:** Input → 64 neurons → 32 neurons → Output
- **Why used:** Can capture complex relationships; sklearn's `MLPClassifier` is the SLP/MLP implementation
- **SLP vs MLP:**
  - SLP = Single Layer Perceptron (no hidden layers, only linear separation)
  - MLP = Multiple hidden layers (can learn non-linear patterns — more powerful)
- **Best for:** Complex patterns where simpler models may fail

---

### 6️⃣ K-Means Clustering (Unsupervised)

- **What it does:** Groups customers into **K clusters** based on feature similarity (without using churn labels)
- **Why used:** To discover natural groupings — e.g., "high-risk" vs "low-risk" customers — without supervision
- **Elbow Method:** Used to choose the best value of K
- **Best for:** Customer segmentation and exploratory analysis

---

### 7️⃣ Hierarchical Clustering (Unsupervised)

- **What it does:** Builds a tree (dendrogram) of clusters from bottom up (agglomerative)
- **Why used:** Doesn't need K specified in advance; shows relationships between data at all levels
- **Dendrogram:** A tree diagram showing how clusters merge — cut at a height to get N clusters
- **Ward method:** Minimizes variance within clusters
- **Best for:** When you don't know how many clusters to expect

---

### 8️⃣ PCA — Principal Component Analysis (Dimensionality Reduction)

- **What it does:** Reduces many features into fewer components while preserving maximum variance
- **Why used:** Our data has 8 features — PCA reduces to 2 dimensions so we can **visualize** it in 2D
- **PC1 and PC2:** New axes that capture the most variance in data
- **Why important:** Helps remove noise, speeds up training, and enables visualization
- **Not a classifier:** It's a preprocessing/visualization technique

---

## 🔑 Key Terms for Viva

| Term | Meaning |
|---|---|
| **Accuracy** | % of correct predictions out of total |
| **Confusion Matrix** | Table showing TP, TN, FP, FN predictions |
| **Precision** | Of all predicted churns, how many were actually churn |
| **Recall** | Of all actual churns, how many did we correctly catch |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Overfitting** | Model performs well on training, poorly on test |
| **Feature Scaling** | Normalizing features so no single one dominates (StandardScaler) |
| **Label Encoding** | Converting text categories to numbers (Male=1, Female=0) |
| **Train-Test Split** | 80% data for training, 20% for testing |
| **Inertia (K-Means)** | Sum of squared distances from points to their cluster center |

---

## 🏆 Which Model is Best and Why?

In this project, **Logistic Regression** or **MLP** typically performs best because:
- The data has a somewhat linear decision boundary
- MLP captures non-linearities with its hidden layers
- Decision Tree may overfit on small datasets

> Always check the accuracy comparison bar chart in your notebook for the final answer in your specific run.

---

## 💡 Summary Flow

```
Raw Data
   ↓
Preprocessing (Encoding + Scaling)
   ↓
EDA (Visualize patterns)
   ↓
┌──────────────────┬───────────────────┐
│  Supervised      │  Unsupervised     │
│  (with labels)   │  (no labels)      │
│  ─────────────   │  ───────────────  │
│  LogReg          │  K-Means          │
│  LinearReg       │  Hierarchical     │
│  DecisionTree    │                   │
│  KNN             │  Dimensionality   │
│  MLP             │  Reduction: PCA   │
└──────────────────┴───────────────────┘
   ↓
Model Comparison + Best Model Selection
   ↓
Predict New Customer Churn
```

---

*Prepared for: CS/AI Semester Project | Topic: Customer Churn Prediction*
