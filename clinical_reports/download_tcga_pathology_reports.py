import requests
import os
import pandas as pd
from tqdm import tqdm

def download_tcga_pathology_reports(csv_path, output_dir="tcga_reports"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return
    
    if 'slide_id' not in df.columns:
        return

    df['case_id'] = df['slide_id'].apply(lambda x: "-".join(str(x).split("-")[:3]) if pd.notnull(x) else None)
    unique_cases = df['case_id'].dropna().unique().tolist()
    
    
    if len(unique_cases) == 0:
        return

    endpoint = "https://api.gdc.cancer.gov/files"
    
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": unique_cases
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Pathology Report"
                }
            }
        ]
    }
    
    payload = {
        "filters": filters,
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": "1000"
    }
    
    response = requests.post(endpoint, json=payload)
    
    if response.status_code != 200:
        print(f"GDC Server Error: {response.status_code}")
        print(f"{response.text}")
        return
        
    try:
        data = response.json()
    except Exception as e:
        return
    
    if "data" not in data or "hits" not in data["data"] or len(data["data"]["hits"]) == 0:
        return
        
    hits = data["data"]["hits"]
    
    os.makedirs(output_dir, exist_ok=True)
    data_endpoint = "https://api.gdc.cancer.gov/data/"
    
    for hit in tqdm(hits, desc="Downloading PDFs"):
        file_id = hit["file_id"]
        file_name = hit["file_name"]
        
        case_id = hit.get("cases", [{}])[0].get("submitter_id", "UNKNOWN")
        
        save_name = f"{case_id}_{file_name}"
        save_path = os.path.join(output_dir, save_name)
        
        if not os.path.exists(save_path):
            file_url = data_endpoint + file_id
            r = requests.get(file_url, stream=True)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
if __name__ == "__main__":
    csv_file_path = '/content/AttriMIL-LungCancer/evaluation_results/extracted_visual_scores.csv'
    download_tcga_pathology_reports(csv_file_path)