import pandas as pd
import numpy as np
import os

def clean_data():

    print("📂 Loading raw dataset...")
    df = pd.read_csv("data/raw/health_raw.csv")

    # -----------------------------
    # 1) Basic Info
    # -----------------------------
    print("\n📊 Dataset Shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)

    # -----------------------------
    # 2) Missing Values
    # -----------------------------
    print("\n❓ Missing values:")
    print(df.isnull().sum())

    # แยก column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # เติมค่า missing
    print("\n🧹 Filling missing values...")

    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # -----------------------------
    # 3) Remove duplicates
    # -----------------------------
    duplicates = df.duplicated().sum()
    print("\nDuplicate rows:", duplicates)

    df.drop_duplicates(inplace=True)

    # -----------------------------
    # 4) Save cleaned data
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)

    save_path = "data/processed/health_cleaned.csv"
    df.to_csv(save_path, index=False)

    print("\n✅ Clean dataset saved to:", save_path)

if __name__ == "__main__":
    clean_data()
