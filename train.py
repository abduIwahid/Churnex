import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import shap

def train_model():
    print("Loading data...")
    df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df_raw.copy()

    # Preprocessing
    print("Preprocessing data...")
    # Drop customerID if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Convert TotalCharges to numeric, impute with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Number of services
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    existing_service_cols = [c for c in service_cols if c in df.columns]
    if existing_service_cols:
        df['num_services'] = (df[existing_service_cols] == 'Yes').sum(axis=1)

    # Rename columns and keep only the ones used for modeling
    rename_map = {
        'SeniorCitizen'  : 'senior_citizen',
        'tenure'         : 'tenure',
        'MonthlyCharges' : 'monthly_charges',
        'TotalCharges'   : 'total_charges',
        'Contract'       : 'contract',
        'InternetService': 'internet',
        'gender'         : 'gender',
        'Churn'          : 'churn',
        'num_services'   : 'num_services'
    }
    keep_cols = [c for c in rename_map if c in df.columns]
    df = df[keep_cols].rename(columns=rename_map)

    # Encode categorical columns and store encoders
    cat_cols = ['contract', 'internet', 'gender']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Handle any remaining NaNs just in case
    X = X.fillna(0)

    # Make sure we keep the same feature order
    feature_names = X.columns.tolist()

    # Scale the features
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Train MLP Model
    print("Training MLP Model on the full dataset...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        max_iter=500,
        activation='relu',
        random_state=42
    )
    mlp.fit(X_scaled, y)
    
    accuracy = mlp.score(X_scaled, y)
    print(f"Training complete. Accuracy on full dataset: {accuracy:.4f}")

    print("Computing SHAP background dataset...")
    # Use kmeans to summarize the dataset for the SHAP explainer
    background = shap.kmeans(X_scaled, 50)

    # Export model and preprocessors
    print("Saving model and preprocessors...")
    model_data = {
        'model': mlp,
        'scaler': sc,
        'encoders': encoders,
        'feature_names': feature_names,
        'shap_background': background
    }
    joblib.dump(model_data, 'model.joblib')
    print("Saved to model.joblib successfully!")

if __name__ == "__main__":
    train_model()
