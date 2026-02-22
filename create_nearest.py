import os
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def main():
    raw_data_dirs = ['/content/extracted_features/TCGA-LUAD', '/content/extracted_features/TCGA-LUSC']
    save_dir = '/content/AttriMIL_Workspace/data/h5_coords_files'
    os.makedirs(save_dir, exist_ok=True)

    print("در حال هماهنگ‌سازی دیتا با استاندارد AttriMIL...")

    for data_dir in raw_data_dirs:
        if not os.path.exists(data_dir): continue
        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        
        for name in tqdm(h5_files):
            try:
                with h5py.File(os.path.join(data_dir, name), 'r') as h5_in:
                    features = np.array(h5_in['features']).squeeze(0)
                    coords = np.array(h5_in['coords']).squeeze(0)
                
                nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(coords)
                _, indices = nbrs.kneighbors(coords)
                nearest = indices[:, 1:]
                
                with h5py.File(os.path.join(save_dir, name), 'w') as h5_out:
                    h5_out.create_dataset('features', data=features)
                    h5_out.create_dataset('coords', data=coords)
                    h5_out.create_dataset('nearest', data=nearest)
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()