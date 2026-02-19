import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_user_segments():

    print("📂 Loading feature dataset...")
    df = pd.read_csv("data/processed/health_features.csv")

    print("Dataset shape:", df.shape)

    # -----------------------------
    # เลือก feature สำหรับ clustering
    # -----------------------------
    features = [
        "Sleep_Health_Score",
        "Activity_Health_Score",
        "Cardiovascular_Health_Score",
        "Mental_Health_Score",
        "Overall_Wellness_Score"
    ]

    X = df[features]

    # -----------------------------
    # Scale data
    # -----------------------------
    print("⚙️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # K-Means Clustering
    # -----------------------------
    print("🤖 Running K-Means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

    df["User_Segment"] = kmeans.fit_predict(X_scaled)

    # ดูจำนวนคนในแต่ละ segment
    print("\nSegment distribution:")
    print(df["User_Segment"].value_counts())

    # -----------------------------
    # Save dataset
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)

    save_path = "data/processed/health_segmented.csv"
    df.to_csv(save_path, index=False)

    print("\n✅ Segmented dataset saved to:", save_path)

if __name__ == "__main__":
    create_user_segments()
