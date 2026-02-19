import pandas as pd
import kagglehub
import os

def download_dataset():
    print("📥 Downloading dataset from Kaggle...")

    # ดาวน์โหลด dataset
    path = kagglehub.dataset_download(
        "mahdimashayekhi/health-and-lifestyle-dataset"
    )

    print("Dataset downloaded at:", path)

    # หาไฟล์ csv
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    print("Found CSV files:", csv_files)

    # โหลด csv
    file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)

    # สร้างโฟลเดอร์ raw ถ้ายังไม่มี
    os.makedirs("data/raw", exist_ok=True)

    # เซฟไฟล์
    save_path = "data/raw/health_raw.csv"
    df.to_csv(save_path, index=False)

    print("✅ Saved raw dataset to:", save_path)

if __name__ == "__main__":
    download_dataset()
