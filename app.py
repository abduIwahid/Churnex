from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import io
import shap_lite as shap
from fpdf import FPDF

# Base directory — always the repo root regardless of CWD on Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Churnex",
    description="AI-powered Customer Churn Intelligence Platform",
    version="1.0.0"
)

# Mount the static files directory using an absolute path
_STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Load the trained model and preprocessors
try:
    model_data = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoders = model_data["encoders"]
    feature_names = model_data["feature_names"]
    shap_background = model_data.get("shap_background", None)
    print("Model loaded successfully.")
    
    # Initialize SHAP explainer
    # We explain the positive class probability
    if shap_background is not None:
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], shap_background)
        print("SHAP explainer initialized.")
    else:
        explainer = None
        print("No SHAP background found. Explainer not initialized.")
        
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    explainer = None

class CustomerData(BaseModel):
    senior_citizen: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    internet: str
    gender: str
    num_services: int

def generate_recommendations(customer: dict, prob: float):
    if prob < 0.5:
        return ["Customer is at low risk. Maintain current service quality."]
    
    recs = []
    if customer.get('contract') == 'Month-to-month':
        recs.append("Offer a 10% discount to upgrade to a 1-year or 2-year contract.")
    
    if customer.get('monthly_charges', 0) > 80:
        recs.append("Monthly charges are high. Consider a tailored bundle to increase perceived value.")
    
    if customer.get('num_services', 0) < 2:
        recs.append("Cross-sell additional services (like Tech Support or Security) with a free trial.")
    
    if not recs:
        recs.append("Reach out for a personalized check-in to ensure satisfaction.")
    return recs

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        cust_dict = customer.dict()
        df = pd.DataFrame([cust_dict])

        # Apply encoders
        for col, le in encoders.items():
            if col in df.columns:
                if df[col][0] not in le.classes_:
                    raise ValueError(f"Invalid value for {col}: {df[col][0]}")
                df[col] = le.transform(df[col])

        # Ensure columns match training order
        df = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(df)

        # Predict
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        # SHAP Explanations
        shap_values_dict = {}
        if explainer is not None:
            shap_vals = explainer.shap_values(X_scaled)
            # shap_vals is usually an array of shape (1, num_features)
            # Map values back to feature names
            for i, name in enumerate(feature_names):
                shap_values_dict[name] = float(shap_vals[0][i])
                
        # Recommendations
        recs = generate_recommendations(cust_dict, probability)

        return {
            "churn_prediction": prediction,
            "churn_probability": probability,
            "message": "High Risk of Churn" if prediction == 1 else "Likely to Retain",
            "shap_values": shap_values_dict,
            "recommendations": recs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_bulk")
async def predict_bulk(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        contents = await file.read()

        # Try UTF-8 (with BOM strip) first, fall back to latin-1 for Excel-exported CSVs
        try:
            text = contents.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = contents.decode('latin-1')

        df_raw = pd.read_csv(io.StringIO(text))

        # Strip whitespace from column names and build case-insensitive lookup
        df_raw.columns = [c.strip() for c in df_raw.columns]
        col_lower = {c.lower(): c for c in df_raw.columns}  # lower → original name

        df = df_raw.copy()

        # ── TELCO FORMAT DETECTION ──────────────────────────────────────────
        # Telco format has: SeniorCitizen / MonthlyCharges / InternetService / Contract
        is_telco = any(k in col_lower for k in ['seniorcitizen', 'monthlycharges', 'internetservice'])

        if is_telco:
            # ── TELCO CSV PATH ──────────────────────────────────────────────
            rename_map = {
                col_lower.get('seniorcitizen',   'SeniorCitizen')   : 'senior_citizen',
                col_lower.get('tenure',          'tenure')           : 'tenure',
                col_lower.get('monthlycharges',  'MonthlyCharges')   : 'monthly_charges',
                col_lower.get('totalcharges',    'TotalCharges')     : 'total_charges',
                col_lower.get('contract',        'Contract')         : 'contract',
                col_lower.get('internetservice', 'InternetService')  : 'internet',
                col_lower.get('gender',          'gender')           : 'gender',
            }
            rename_map = {k: v for k, v in rename_map.items() if k in df.columns}

            # Calculate num_services from individual service columns
            service_cols_lower = ['phoneservice', 'multiplelines', 'onlinesecurity',
                                   'onlinebackup', 'deviceprotection', 'techsupport',
                                   'streamingtv', 'streamingmovies']
            existing_svc = [col_lower[lc] for lc in service_cols_lower if lc in col_lower]
            if existing_svc and 'num_services' not in col_lower:
                df['num_services'] = (df[existing_svc] == 'Yes').sum(axis=1)

            df = df.rename(columns=rename_map)

            # Convert TotalCharges (may contain blanks)
            if 'total_charges' in df.columns:
                df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
                med = df['total_charges'].median()
                df['total_charges'] = df['total_charges'].fillna(0 if pd.isna(med) else med)

        else:
            # ── ALTERNATIVE CSV PATH ────────────────────────────────────────
            # Supports columns: Age, Gender, Tenure, Usage Frequency,
            # Support Calls, Payment Delay, Subscription Type,
            # Contract Length, Total Spend, Last Interaction

            # senior_citizen: Age >= 60 → 1, else 0
            age_col = col_lower.get('age')
            if age_col:
                df['senior_citizen'] = (pd.to_numeric(df[age_col], errors='coerce').fillna(0) >= 60).astype(int)
            else:
                df['senior_citizen'] = 0

            # gender: keep as-is or map to Male/Female
            gender_col = col_lower.get('gender')
            if gender_col and gender_col != 'gender':
                df = df.rename(columns={gender_col: 'gender'})
            elif not gender_col:
                df['gender'] = 'Male'

            # tenure: map directly
            tenure_col = col_lower.get('tenure')
            if tenure_col and tenure_col != 'tenure':
                df = df.rename(columns={tenure_col: 'tenure'})
            elif not tenure_col:
                df['tenure'] = 0

            # total_charges: map from 'Total Spend'
            tc_col = col_lower.get('total spend') or col_lower.get('totalspend') or col_lower.get('total_spend')
            if tc_col:
                df['total_charges'] = pd.to_numeric(df[tc_col], errors='coerce').fillna(0)
            else:
                df['total_charges'] = 0

            # monthly_charges: derive from total_charges / max(tenure, 1)
            ten_series = pd.to_numeric(df['tenure'], errors='coerce').fillna(1).replace(0, 1)
            df['monthly_charges'] = (df['total_charges'] / ten_series).round(2)

            # contract: map 'Contract Length' → Monthly/One year/Two year
            cl_col = col_lower.get('contract length') or col_lower.get('contractlength') or col_lower.get('contract_length')
            if cl_col:
                def map_contract(val):
                    v = str(val).strip().lower()
                    if v in ('monthly', 'month', 'month-to-month', '1'):
                        return 'Month-to-month'
                    elif v in ('annual', 'one year', 'oneyear', '1 year', '12', '2'):
                        return 'One year'
                    elif v in ('two year', 'twoyear', '2 year', '24', 'bi-annual', '3'):
                        return 'Two year'
                    return 'Month-to-month'
                df['contract'] = df[cl_col].apply(map_contract)
            else:
                df['contract'] = 'Month-to-month'

            # internet: map 'Subscription Type' → DSL / Fiber optic / No
            sub_col = col_lower.get('subscription type') or col_lower.get('subscriptiontype') or col_lower.get('subscription_type')
            if sub_col:
                def map_internet(val):
                    v = str(val).strip().lower()
                    if 'basic' in v:
                        return 'DSL'
                    elif 'premium' in v or 'standard' in v:
                        return 'Fiber optic'
                    elif 'none' in v or 'no' in v:
                        return 'No'
                    return 'DSL'
                df['internet'] = df[sub_col].apply(map_internet)
            else:
                df['internet'] = 'DSL'

            # num_services: approximate from Support Calls + Usage Frequency (capped 0-8)
            sc_col  = col_lower.get('support calls')  or col_lower.get('supportcalls')
            uf_col  = col_lower.get('usage frequency') or col_lower.get('usagefrequency')
            if sc_col or uf_col:
                sc_vals = pd.to_numeric(df[sc_col],  errors='coerce').fillna(0) if sc_col  else 0
                uf_vals = pd.to_numeric(df[uf_col],  errors='coerce').fillna(0) if uf_col  else 0
                df['num_services'] = ((sc_vals + uf_vals) / 2).clip(0, 8).round().astype(int)
            else:
                df['num_services'] = 4  # neutral default

        # ── COMMON: validate all features are present ───────────────────────
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Could not map all required model features. Still missing: {missing}. "
                f"Columns in your file: {list(df_raw.columns)}."
            )

        df_clean = df[feature_names].copy()

        # Encode categorical columns
        for col, le in encoders.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).map(
                    lambda s, le=le: s if s in le.classes_ else le.classes_[0]
                )
                df_clean[col] = le.transform(df_clean[col])

        # Fill any remaining NaN
        df_clean = df_clean.fillna(0)

        # Scale & Predict
        X_scaled = scaler.transform(df_clean)
        preds    = model.predict(X_scaled)
        probs    = model.predict_proba(X_scaled)[:, 1]

        # Build result dataframe
        df_res = df_raw.copy()
        df_res['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in preds]
        df_res['Churn_Probability'] = [round(float(p), 4) for p in probs]

        stream = io.StringIO()
        df_res.to_csv(stream, index=False)

        return Response(
            content=stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.get("/api/stats")
def get_stats():
    try:
        # Load the original raw dataset for stats
        df = pd.read_csv(os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
        
        total_customers = len(df)
        churned = len(df[df['Churn'] == 'Yes'])
        churn_rate = churned / total_customers * 100
        
        contract_counts = df['Contract'].value_counts().to_dict()
        
        # Average monthly charges by churn
        avg_charges = df.groupby('Churn')['MonthlyCharges'].mean().to_dict()
        
        return {
            "total_customers": total_customers,
            "churn_rate": round(churn_rate, 2),
            "contract_distribution": contract_counts,
            "avg_monthly_charges": avg_charges
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export_pdf")
def export_pdf(customer: CustomerData):
    try:
        # Generate prediction first
        cust_dict = customer.dict()
        df = pd.DataFrame([cust_dict])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        df = df[feature_names]
        X_scaled = scaler.transform(df)
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])
        recs = generate_recommendations(cust_dict, probability)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Churnex - Customer Churn Prediction Report", ln=True, align='C')
        pdf.ln(10)
        
        # Customer Profile
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Customer Profile:", ln=True)
        pdf.set_font("Arial", '', 11)
        for k, v in cust_dict.items():
            pdf.cell(200, 8, txt=f"  {k.replace('_', ' ').title()}: {v}", ln=True)
        pdf.ln(5)
        
        # Prediction
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Prediction Details:", ln=True)
        pdf.set_font("Arial", '', 11)
        status = "High Risk of Churn" if prediction == 1 else "Likely to Retain"
        pdf.cell(200, 8, txt=f"  Status: {status}", ln=True)
        pdf.cell(200, 8, txt=f"  Churn Probability: {probability*100:.1f}%", ln=True)
        pdf.ln(5)
        
        # Recommendations
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Recommended Actions:", ln=True)
        pdf.set_font("Arial", '', 11)
        for rec in recs:
            pdf.cell(200, 8, txt=f"  - {rec}", ln=True)
            
        pdf_bytes = bytes(pdf.output())
        
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
