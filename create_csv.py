import os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create CSV label file for pathology slides")
    
    parser.add_argument('--luad_dir', type=str, 
                        default='/content/extracted_features/TCGA-LUAD',
                        help='Path to extracted features for LUAD class (Label 0)')
    
    parser.add_argument('--lusc_dir', type=str, 
                        default='/content/extracted_features/TCGA-LUSC',
                        help='Path to extracted features for LUSC class (Label 1)')
    
    parser.add_argument('--csv_save_path', type=str, 
                        default='/content/AttriMIL-LungCancer/datasets/tcga_nsclc_labels.csv',
                        help='Path and filename for the output CSV')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    luad_dir = args.luad_dir
    lusc_dir = args.lusc_dir
    csv_save_path = args.csv_save_path
    
    csv_dir = os.path.dirname(csv_save_path)
    if csv_dir: 
        os.makedirs(csv_dir, exist_ok=True)

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
    
    if len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(csv_save_path, index=False)
        print(f"\n✅ Label file successfully created with {len(data)} records:\n💾 {csv_save_path}")
    else:
        print("\n❌ No .h5 files found in the specified directories! CSV was not created.")

if __name__ == "__main__":
    main()