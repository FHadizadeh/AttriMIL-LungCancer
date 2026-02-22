import os
import pandas as pd

def main():
    luad_dir = '/content/extracted_features/TCGA-LUAD'
    lusc_dir = '/content/extracted_features/TCGA-LUSC'
    
    csv_dir = '/content/AttriMIL-LungCancer/datasets'
    os.makedirs(csv_dir, exist_ok=True)
    csv_save_path = os.path.join(csv_dir, 'tcga_nsclc_labels.csv')

    data = []

    if os.path.exists(luad_dir):
        for file in os.listdir(luad_dir):
            if file.endswith('.h5'):
                slide_id = file.replace('.h5', '')
                case_id = slide_id[:12] 
                data.append({'case_id': case_id, 'slide_id': slide_id, 'label': 0})

    if os.path.exists(lusc_dir):
        for file in os.listdir(lusc_dir):
            if file.endswith('.h5'):
                slide_id = file.replace('.h5', '')
                case_id = slide_id[:12]
                data.append({'case_id': case_id, 'slide_id': slide_id, 'label': 1})

    df = pd.DataFrame(data)
    df.to_csv(csv_save_path, index=False)

if __name__ == "__main__":
    main()