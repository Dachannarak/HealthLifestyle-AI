import pandas as pd
import numpy as np
import os

def create_health_scores():

    print("📂 Loading cleaned dataset...")
    df = pd.read_csv("data/processed/health_cleaned.csv")

    print("Dataset shape:", df.shape)

    # -------------------------------------------------
    # 1️⃣ Sleep Health Score (0-100)
    # ideal sleep = 7-9 hours
    # -------------------------------------------------
    print("🛌 Creating Sleep Health Score...")

    df["Sleep_Health_Score"] = 100 - abs((df["Hours_of_Sleep"] - 8) * 12.5)
    df["Sleep_Health_Score"] = np.clip(df["Sleep_Health_Score"], 0, 100)

    # -------------------------------------------------
    # 2️⃣ Activity Health Score
    # based on exercise + steps
    # -------------------------------------------------
    print("🏃 Creating Activity Health Score...")

    exercise_score = (df["Exercise_Hours_per_Week"] / 10) * 100
    steps_score = (df["Daily_Steps"] / 10000) * 100

    df["Activity_Health_Score"] = (exercise_score * 0.6 + steps_score * 0.4)
    df["Activity_Health_Score"] = np.clip(df["Activity_Health_Score"], 0, 100)

    # -------------------------------------------------
    # 3️⃣ Cardiovascular Health Score
    # ideal resting heart rate ≈ 70 bpm
    # -------------------------------------------------
    print("❤️ Creating Cardiovascular Health Score...")

    df["Cardiovascular_Health_Score"] = 100 - abs(df["Heart_Rate"] - 70)
    df["Cardiovascular_Health_Score"] = np.clip(df["Cardiovascular_Health_Score"], 0, 100)

    # -------------------------------------------------
    # 4️⃣ Mental Health Score (synthetic)
    # -------------------------------------------------
    print("🧠 Creating Mental Health Score...")

    np.random.seed(42)
    df["Mental_Health_Score"] = np.random.uniform(50, 100, len(df))

    # -------------------------------------------------
    # 5️⃣ Overall Wellness Score
    # -------------------------------------------------
    print("🌟 Creating Overall Wellness Score...")

    df["Overall_Wellness_Score"] = df[
        [
            "Sleep_Health_Score",
            "Activity_Health_Score",
            "Cardiovascular_Health_Score",
            "Mental_Health_Score",
        ]
    ].mean(axis=1)

    # -------------------------------------------------
    # Save dataset
    # -------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)

    save_path = "data/processed/health_features.csv"
    df.to_csv(save_path, index=False)

    print("\n✅ Feature dataset saved to:", save_path)

if __name__ == "__main__":
    create_health_scores()
