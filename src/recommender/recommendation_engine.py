import pandas as pd
import os

# --------------------------------------------------
# 🏋️ Exercise Recommendation
# --------------------------------------------------
def get_exercise_recommendation(row):

    score = row["Activity_Health_Score"]
    bmi = row["BMI"]

    if score >= 75:
        rec = "HIIT, Running, Weight Training"
    elif score >= 50:
        rec = "Cycling, Swimming, Gym"
    else:
        rec = "Walking, Yoga, Stretching"

    if bmi >= 30:
        rec += " (Low-impact focus)"

    return rec


# --------------------------------------------------
# 🍎 Nutrition Recommendation
# --------------------------------------------------
def get_nutrition_recommendation(row):

    bmi = row["BMI"]

    if bmi >= 30:
        return "Reduce sugar, eat vegetables, portion control"
    elif bmi < 20:
        return "Increase calories, more protein"
    else:
        return "Balanced diet"


# --------------------------------------------------
# 😴 Lifestyle Recommendation
# --------------------------------------------------
def get_lifestyle_recommendation(row):

    sleep = row["Hours_of_Sleep"]

    if sleep < 6:
        return "Increase sleep to 7-9 hours"
    elif sleep > 9:
        return "Avoid oversleeping"
    else:
        return "Good sleep habits"


# --------------------------------------------------
# 🚀 MAIN FUNCTION
# --------------------------------------------------
def generate_recommendations():

    print("📂 Loading segmented dataset...")
    df = pd.read_csv("data/processed/health_segmented.csv")

    print("Generating recommendations...")

    df["Exercise_Recommendation"] = df.apply(get_exercise_recommendation, axis=1)
    df["Nutrition_Recommendation"] = df.apply(get_nutrition_recommendation, axis=1)
    df["Lifestyle_Recommendation"] = df.apply(get_lifestyle_recommendation, axis=1)

    # Save output
    os.makedirs("data/output", exist_ok=True)

    save_path = "data/output/health_recommendations.csv"
    df.to_csv(save_path, index=False)

    print("\n✅ Recommendations saved to:", save_path)

if __name__ == "__main__":
    generate_recommendations()
