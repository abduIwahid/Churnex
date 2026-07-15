# Churnex 🔮

> **AI-Powered Customer Churn Intelligence Platform**

Churnex is an end-to-end machine learning web application that predicts customer churn for telecom companies. It combines a trained **Multi-Layer Perceptron (MLP) neural network** with a **FastAPI** backend and a sleek dark-themed frontend to deliver real-time churn risk assessments, explainable AI insights via **SHAP values**, actionable retention recommendations, and batch CSV processing — all from a single-page web interface.

---

## 📸 Features at a Glance

| Feature | Description |
|---|---|
| 🎯 **Single Prediction** | Enter a customer's profile and get an instant churn probability with SHAP explanation |
| 📁 **Bulk CSV Prediction** | Upload any CSV (Telco format or alternative) — results are auto-mapped and returned as a downloadable file |
| 📊 **Analytics Dashboard** | Live statistics from the training dataset: total customers, churn rate, contract distribution, and charge comparisons |
| 🧠 **Explainable AI (SHAP)** | Butterfly-style bar charts showing which features push the prediction toward or away from churn |
| 💡 **Retention Recommendations** | Business-rule-driven, context-aware retention actions generated per prediction |
| 📄 **PDF Export** | One-click PDF report generation containing the customer profile, prediction, and recommendations |

---

## 🗂️ Project Structure

```
Churnex/
├── app.py                              # FastAPI backend (all API routes)
├── train.py                            # Model training pipeline
├── model.joblib                        # Serialized model + preprocessors (auto-generated)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # IBM Telco dataset (training + dashboard stats)
├── requirements.txt                    # Python dependencies
├── VIVA_Explanation.md                 # Academic ML explanation reference
├── Customer_Churn_Prediction_Updated.ipynb  # Jupyter notebook (EDA + all models)
└── static/
    ├── index.html                      # Single-page frontend
    ├── style.css                       # Dark glassmorphism UI styles
    └── script.js                       # Frontend logic (fetch, charts, SHAP rendering)
```

---

## 🧠 Machine Learning Pipeline

### Dataset
- **IBM Telco Customer Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- ~7,000 customer records with 21 features
- Binary target: `Churn` (Yes / No)

### Feature Engineering (`train.py`)
The training pipeline selects and engineers **8 core features**:

| Feature | Description |
|---|---|
| `senior_citizen` | Whether the customer is a senior citizen (0/1) |
| `tenure` | Number of months the customer has been with the company |
| `monthly_charges` | The amount charged per month |
| `total_charges` | Total amount charged over the customer's lifetime |
| `contract` | Contract type — Month-to-month / One year / Two year |
| `internet` | Internet service type — DSL / Fiber optic / No |
| `gender` | Customer gender — Male / Female |
| `num_services` | Number of add-on services subscribed (0–8) |

### Preprocessing
1. **Drop** `customerID` (non-informative identifier)
2. **Impute** `TotalCharges` blanks with the column median
3. **Engineer** `num_services` by counting "Yes" across 8 service columns
4. **Label Encode** categorical features (`contract`, `internet`, `gender`)
5. **Standard Scale** all features via `StandardScaler`

### Model: Multi-Layer Perceptron (MLP)
```
Architecture:  Input(8) → Dense(128) → Dense(64) → Dense(32) → Dense(16) → Output(1)
Activation:    ReLU
Max Iterations: 500
Random State:  42
```
The model is trained on the **full dataset** (no train/test split) to maximize generalization for production inference.

### Explainability: SHAP
After training, a **SHAP KernelExplainer** background is computed using `shap.kmeans(X_scaled, 50)` — a 50-sample KMeans summary of the training data. This is saved into `model.joblib` alongside the model, scaler, and encoders, so SHAP explanations are available at inference time with **zero retraining**.

### Saved Artifacts (`model.joblib`)
```python
{
    "model":           MLPClassifier,
    "scaler":          StandardScaler,
    "encoders":        { "contract": LabelEncoder, "internet": LabelEncoder, "gender": LabelEncoder },
    "feature_names":   ["senior_citizen", "tenure", ...],
    "shap_background": shap.KMeansData  # 50 cluster centers
}
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.8+**
- pip

### 1. Clone / Navigate to the Project
```bash
cd Churnex
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt` installs:**
```
fastapi
uvicorn
pandas
scikit-learn
joblib
numpy
shap
python-multipart
fpdf2
```

### 3. Train the Model (first time only)
> Skip this step if `model.joblib` already exists in the project root.

```bash
python train.py
```

This will:
- Load and preprocess `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Train the MLP classifier on the full dataset
- Compute the SHAP background
- Save everything to `model.joblib`

Expected output:
```
Loading data...
Preprocessing data...
Training MLP Model on the full dataset...
Training complete. Accuracy on full dataset: 0.xxxx
Computing SHAP background dataset...
Saving model and preprocessors...
Saved to model.joblib successfully!
```

### 4. Run the Application
```bash
uvicorn app:app --reload
```

Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔌 API Reference

All endpoints are served by **FastAPI**. Interactive docs are auto-generated at `/docs` (Swagger UI) and `/redoc`.

---

### `GET /`
**Returns:** The frontend `index.html` single-page application.

---

### `POST /predict`
**Single customer churn prediction.**

**Request Body (JSON):**
```json
{
  "senior_citizen":    0,
  "tenure":            24,
  "monthly_charges":   75.50,
  "total_charges":     1812.00,
  "contract":          "Month-to-month",
  "internet":          "Fiber optic",
  "gender":            "Male",
  "num_services":      2
}
```

**Response (JSON):**
```json
{
  "churn_prediction":  1,
  "churn_probability": 0.8234,
  "message":           "High Risk of Churn",
  "shap_values": {
    "senior_citizen":   0.01,
    "tenure":          -0.18,
    "monthly_charges":  0.22,
    "total_charges":   -0.05,
    "contract":         0.31,
    "internet":         0.12,
    "gender":          -0.02,
    "num_services":    -0.07
  },
  "recommendations": [
    "Offer a 10% discount to upgrade to a 1-year or 2-year contract.",
    "Monthly charges are high. Consider a tailored bundle to increase perceived value."
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `churn_prediction` | `int` | `1` = Churn, `0` = Retain |
| `churn_probability` | `float` | Probability of churn (0.0 – 1.0) |
| `message` | `str` | Human-readable verdict |
| `shap_values` | `object` | Per-feature SHAP impact values |
| `recommendations` | `array` | Business action items |

---

### `POST /predict_bulk`
**Batch prediction from a CSV upload.**

**Request:** `multipart/form-data` with a `file` field containing a `.csv` file.

**Supported CSV Formats:**

#### Format A — IBM Telco Format (auto-detected)
Detected when the file contains columns like `SeniorCitizen`, `MonthlyCharges`, or `InternetService`.

| Telco Column | Maps To |
|---|---|
| `SeniorCitizen` | `senior_citizen` |
| `tenure` | `tenure` |
| `MonthlyCharges` | `monthly_charges` |
| `TotalCharges` | `total_charges` |
| `Contract` | `contract` |
| `InternetService` | `internet` |
| `gender` | `gender` |
| `PhoneService` + 7 others | `num_services` (counted) |

#### Format B — Alternative Format
Mapped for generic customer datasets with columns like `Age`, `Contract Length`, `Subscription Type`, `Total Spend`.

| Alternative Column | Maps To | Logic |
|---|---|---|
| `Age` | `senior_citizen` | Age ≥ 60 → 1, else 0 |
| `Gender` | `gender` | Direct map |
| `Tenure` | `tenure` | Direct map |
| `Total Spend` | `total_charges` | Direct map |
| *(derived)* | `monthly_charges` | `total_charges / max(tenure, 1)` |
| `Contract Length` | `contract` | `monthly/annual/bi-annual` → `Month-to-month/One year/Two year` |
| `Subscription Type` | `internet` | `basic→DSL`, `premium/standard→Fiber optic`, `none→No` |
| `Support Calls` + `Usage Frequency` | `num_services` | `(sc + uf) / 2`, clipped 0–8 |

**Response:** A downloadable `predictions.csv` — the original file with two appended columns:
- `Churn_Prediction` — `Yes` or `No`
- `Churn_Probability` — Float (0.0000 – 1.0000)

---

### `GET /api/stats`
**Dataset-level analytics for the Dashboard tab.**

**Response (JSON):**
```json
{
  "total_customers":       7043,
  "churn_rate":            26.54,
  "contract_distribution": { "Month-to-month": 3875, "Two year": 1695, "One year": 1473 },
  "avg_monthly_charges":   { "No": 61.27, "Yes": 74.44 }
}
```

---

### `POST /api/export_pdf`
**Generates and downloads a PDF churn report.**

**Request Body:** Same JSON schema as `/predict`.

**Response:** Binary PDF file download (`report.pdf`).

The PDF includes:
- Customer profile (all 8 input fields)
- Prediction status and churn probability percentage
- Actionable recommendations list

---

## 💡 Retention Recommendation Logic

The `generate_recommendations()` function applies business rules to high-risk customers (`probability ≥ 0.5`):

| Condition | Recommendation |
|---|---|
| `contract == "Month-to-month"` | Offer 10% discount for annual contract upgrade |
| `monthly_charges > $80` | Suggest a tailored bundle to increase perceived value |
| `num_services < 2` | Cross-sell Tech Support or Security with a free trial |
| *(none of the above)* | Personalized check-in call |
| `probability < 0.5` | "Customer is at low risk. Maintain current service quality." |

---

## 🖥️ Frontend Overview

The single-page UI (`static/`) is built with **vanilla HTML, CSS, and JavaScript** — no framework required.

### Tabs
| Tab | What It Does |
|---|---|
| **Single Predict** | Form with 8 fields → calls `/predict` → renders result, SHAP bars, recommendations |
| **Bulk Predict** | Drag-and-drop CSV upload → calls `/predict_bulk` → auto-downloads result CSV |
| **Dashboard** | Calls `/api/stats` (lazy-loaded on first visit) → renders two Chart.js charts |

### SHAP Visualization
Results are rendered as a **butterfly (diverging) bar chart**:
- 🔴 **Red bars** (right side) = features **increasing** churn probability
- 🟢 **Green bars** (left side) = features **decreasing** churn probability
- Features with `|SHAP| < 0.01` are hidden to reduce noise

### Libraries Used (CDN)
- [Chart.js](https://www.chartjs.org/) — Dashboard bar and doughnut charts

---

## 📓 Jupyter Notebook

`Customer_Churn_Prediction_Updated.ipynb` contains the full academic analysis:
- Exploratory Data Analysis (EDA) with visualizations
- Implementation and comparison of **7 ML algorithms**:
  - Logistic Regression
  - Linear Regression (as classifier)
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP) ← **deployed model**
  - K-Means Clustering (unsupervised)
  - Hierarchical Clustering (unsupervised)
- PCA for dimensionality reduction and 2D visualization
- Accuracy comparison bar charts

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, FastAPI, Uvicorn |
| **ML / AI** | scikit-learn (MLP, StandardScaler, LabelEncoder), SHAP, NumPy, pandas |
| **PDF Generation** | fpdf2 |
| **Model Persistence** | joblib |
| **Frontend** | HTML5, CSS3 (glassmorphism), Vanilla JavaScript |
| **Charts** | Chart.js (CDN) |
| **Dataset** | IBM Telco Customer Churn (Kaggle) |

---

## ⚙️ Configuration & Environment

| Variable | Default | Notes |
|---|---|---|
| Host | `127.0.0.1` | Pass `--host 0.0.0.0` to expose on network |
| Port | `8000` | Pass `--port <PORT>` to change |
| Model path | `model.joblib` | Must be in the working directory |
| Dataset path | `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Required for `/api/stats` and retraining |

Example: run on all interfaces at port 5000:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

---

## 🔄 Retraining the Model

If you modify `train.py` or want to retrain from scratch:

```bash
python train.py
```

Then restart the server:
```bash
uvicorn app:app --reload
```

> **Note:** The SHAP background computation (`shap.kmeans`) can take 1–3 minutes. This is a one-time cost at training time; inference is fast.

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `Model is not loaded` (500 error) | Run `python train.py` to generate `model.joblib` |
| `Could not map all required model features` | Ensure your CSV has the required columns (see Bulk Predict section) |
| `TotalCharges` conversion errors | The column may have blank strings — the pipeline handles this with median imputation |
| SHAP values all zero | The `shap_background` key may be missing in an old `model.joblib` — retrain to fix |
| CSV encoding errors | Supported encodings: UTF-8 (with BOM), Latin-1 — other encodings may fail |
| Port 8000 already in use | Run with `--port 8001` or kill the existing process |

---

## 📚 Academic Reference

This project was built as a **CS/AI Semester Project** covering:
- Supervised Learning (classification)
- Unsupervised Learning (clustering)
- Dimensionality Reduction (PCA)
- Neural Networks (MLP)
- Explainable AI (SHAP)

See [`VIVA_Explanation.md`](VIVA_Explanation.md) for a detailed academic breakdown of every algorithm used.

---

## 📄 License

This project is for educational purposes. The dataset is publicly available from IBM via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

*Built with ❤️ using FastAPI + scikit-learn + SHAP*
