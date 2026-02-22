import pandas as pd
import numpy as np
import os

def generate_splits():
    input_csv = '/content/AttriMIL-LungCancer/datasets/tcga_nsclc_labels.csv'
    output_dir = '/content/AttriMIL-LungCancer/splits'
    output_file = os.path.join(output_dir, 'splits_0.csv')

    if not os.path.exists(input_csv):
        print(f"❌ خطا: فایل ورودی در مسیر {input_csv} یافت نشد.")
        return

    df = pd.read_csv(input_csv)
    slides = df['slide_id'].values
    
    np.random.seed(42)
    np.random.shuffle(slides)

    train_idx = int(len(slides) * 0.8)
    val_idx = int(len(slides) * 0.9)

    train_slides = pd.Series(slides[:train_idx])
    val_slides = pd.Series(slides[train_idx:val_idx])
    test_slides = pd.Series(slides[val_idx:])

    split_df = pd.DataFrame({
        'train': train_slides,
        'val': val_slides,
        'test': test_slides
    })

    os.makedirs(output_dir, exist_ok=True)
    split_df.to_csv(output_file, index=False)
    
    

if __name__ == "__main__":
    generate_splits()