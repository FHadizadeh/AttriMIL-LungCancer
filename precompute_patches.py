import os
import glob
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from models.AttriMIL import AttriMIL


def precompute_all_slides(h5_dir, model_weights, output_csv, k=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttriMIL()

    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
    
    all_data = []
    
    for h5_path in tqdm(h5_files):
        slide_id = os.path.basename(h5_path).replace('.h5', '')
        
        with h5py.File(h5_path, 'r') as f:
            features = torch.tensor(f['features'][:], dtype=torch.float32).to(device)
            coords = f['coords'][:]
            
        with torch.no_grad():
            _, _, _, attribute_score, _ = model(features)
            scores = attribute_score.squeeze() 
            
        actual_k = min(k, scores.shape[1])
        
        for class_idx, class_name in enumerate(['LUAD', 'LUSC']):
            class_scores = scores[class_idx]
            top_v, top_i = torch.topk(class_scores, actual_k, largest=True)
            bot_v, bot_i = torch.topk(class_scores, actual_k, largest=False)
            
            for i in range(actual_k):
                # TOP
                all_data.append({'slide_id': slide_id, 'class_branch': class_name,
                                 'type': 'TOP', 'rank': i+1, 'score': top_v[i].item(), 
                                 'coord_x': coords[top_i[i]][0], 'coord_y': coords[top_i[i]][1]})
                # BOTTOM
                all_data.append({'slide_id': slide_id, 'class_branch': class_name,
                                 'type': 'BOTTOM', 'rank': i+1, 'score': bot_v[i].item(), 
                                 'coord_x': coords[bot_i[i]][0], 'coord_y': coords[bot_i[i]][1]})
                
    df_out = pd.DataFrame(all_data)
    df_out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    H5_DIR = "/content/AttriMIL_Workspace/data/h5_coords_files"
    WEIGHTS = "/content/drive/MyDrive/AttriMIL_Backup/save_weights/tcga_nsclc_100/s_0_checkpoint.pt"
    OUT_CSV = "/content/AttriMIL-LungCancer/evaluation_results/master_patch_database.csv"
    
    precompute_all_slides(H5_DIR, WEIGHTS, OUT_CSV)