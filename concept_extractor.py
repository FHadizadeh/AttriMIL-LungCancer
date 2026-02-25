import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import openslide


def auto_download_slide(slide_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{slide_id}.svs")
    
    if os.path.exists(save_path):
        return save_path
            
    file_name = f"{slide_id}.svs"
    payload = {
        "filters": json.dumps({
            "op": "=",
            "content": {"field": "file_name", "value": file_name}
        }),
        "fields": "file_id",
        "format": "JSON"
    }
    
    file_uuid = None
    for endpoint in ["https://api.gdc.cancer.gov/files", "https://api.gdc.cancer.gov/legacy/files"]:
        try:
            r = requests.get(endpoint, params=payload)
            r.raise_for_status()
            data = r.json()
            if data.get("data", {}).get("hits"):
                file_uuid = data["data"]["hits"][0]["file_id"]
                break
        except Exception:
            continue
            
    if not file_uuid:
        return None
        
    
    dl_url = f"https://api.gdc.cancer.gov/data/{file_uuid}"
    try:
        response = requests.get(dl_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
        return save_path
    except Exception as e:
        if os.path.exists(save_path): os.remove(save_path)
        return None 

def render_concept(concept, alignment_csv, master_db_csv, wsi_dir, output_dir, auto_dl=False):
    df_align = pd.read_csv(alignment_csv)
    concept_lower = concept.lower()
    
    if concept_lower not in df_align.columns:
        return
        
    best_slide_row = df_align.sort_values(by=concept_lower, ascending=False).iloc[0]
    best_slide_id = best_slide_row['slide_id']
    
    raw_label = best_slide_row['true_label']
    label_map = {0: 'LUAD', 1: 'LUSC', '0': 'LUAD', '1': 'LUSC'}
    true_label = label_map.get(raw_label, raw_label) 
    
    
    df_db = pd.read_csv(master_db_csv)
    
    slide_patches = df_db[(df_db['slide_id'] == best_slide_id) & (df_db['class_branch'] == true_label)]
    
    if slide_patches.empty:
        return

    wsi_path = os.path.join(wsi_dir, f"{best_slide_id}.svs")
    if not os.path.exists(wsi_path) and auto_dl:
        wsi_path = auto_download_slide(best_slide_id, wsi_dir)
        
    if not wsi_path or not os.path.exists(wsi_path):
        return
        
    wsi = openslide.OpenSlide(wsi_path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for row_idx, p_type in enumerate(['TOP', 'BOTTOM']):
        sub_group = slide_patches[slide_patches['type'] == p_type].sort_values('rank')
        for col_idx in range(min(3, len(sub_group))):
            p_data = sub_group.iloc[col_idx]
            x, y = int(p_data['coord_x']), int(p_data['coord_y'])
            patch = wsi.read_region((x, y), 0, (512, 512)).convert('RGB')
            axes[row_idx, col_idx].imshow(patch)
            axes[row_idx, col_idx].set_title(f"{p_type} Rank {p_data['rank']}\nScore: {p_data['score']:.4f}")
            axes[row_idx, col_idx].axis('off')
            
    plt.suptitle(f"Slide: {best_slide_id[:15]} | Concept: {concept}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{best_slide_id[:15]}_{concept}_vis.png")
    plt.savefig(save_path)
    wsi.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--auto_download", action="store_true")
    args = parser.parse_args()
    
    ALIGN_CSV = "/content/AttriMIL-LungCancer/evaluation_results/ultimate_alignment_data.csv"
    MASTER_DB = "/content/AttriMIL-LungCancer/evaluation_results/master_patch_database.csv"
    WSI_DIR = "/content/wsis/"
    OUT_DIR = "/content/AttriMIL-LungCancer/evaluation_results/visualizations/"
    
    render_concept(args.concept, ALIGN_CSV, MASTER_DB, WSI_DIR, OUT_DIR, args.auto_download)