import os
import re
import pandas as pd
import PyPDF2
from tqdm import tqdm

def extract_text_from_pdfs(reports_dir):
    reports = {}
    for filename in tqdm(os.listdir(reports_dir)):
        if filename.upper().endswith(".PDF"):
            case_id = filename.split('_')[0]
            filepath = os.path.join(reports_dir, filename)
            text = ""
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + " "
            except Exception:
                continue
            
            if text.strip():
                clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
                reports[case_id] = clean_text
    return reports

def is_negated(feature_pattern, raw_text):
    negation_words = r'(no|not|without|absence of|free of|negative for|non)'
    
    pattern = r'\b' + negation_words + r'\b(?:\s+\w+){0,4}\s*(' + feature_pattern + r')\b'
    hyphen_pattern = r'\b(non|un)-(' + feature_pattern + r')\b'
    
    if re.search(pattern, raw_text) or re.search(hyphen_pattern, raw_text):
        return True
    return False

def build_ultimate_semantic_matrix(visual_scores_path, reports_dir, output_csv_path):
    reports_data = extract_text_from_pdfs(reports_dir)
    if not reports_data:
        return

    # Concept Normalization & Semantic Grouping
    # کلید = اسم ستون نهایی در CSV | مقدار = عبارات منظم (Regex) شامل کلمات و مخفف‌ها
    TARGET_FEATURES_MAPPING = {
    'glandular_formation': r'glandular|glands|acinar|acini',
    'adenocarcinoma': r'adenocarcinoma|aca|nsclc',
    'squamous': r'squamous|sqcc|scc',
    'keratinization': r'keratinization|keratin',
    'mucin': r'mucin|mucinous',
    'papillary': r'papillary',
    'lepidic': r'lepidic',
    'necrosis': r'necrosis|necrotic',
    'basaloid': r'basaloid',
    'pleomorphism': r'pleomorphism|pleomorphic',
    'mitosis': r'mitosis|mitotic',
    'differentiation': r'differentiation|differentiated|poorly|moderately', 
    'invasion': r'invasion|invasive|extension', 
    'vascular_lymphatic': r'vascular|lvi|lymphatic', 
    'metastasis': r'metastasis|metastatic', 
    'peribronchial': r'peribronchial|bronchus' 
}

    
    df = pd.read_csv(visual_scores_path)
    df['case_id'] = df['slide_id'].apply(lambda x: "-".join(str(x).split("-")[:3]))
    
    for concept_name in TARGET_FEATURES_MAPPING.keys():
        df[concept_name] = 0
        df[f"absence_of_{concept_name}"] = 0
        
    for index, row in df.iterrows():
        case_id = row['case_id']
        if case_id in reports_data:
            text = reports_data[case_id]
            
            for concept_name, regex_pattern in TARGET_FEATURES_MAPPING.items():
                if re.search(r'\b(' + regex_pattern + r')\b', text):
                    if is_negated(regex_pattern, text):
                        df.at[index, f"absence_of_{concept_name}"] = 1
                    else:
                        df.at[index, concept_name] = 1
                        
    cols_to_keep = ['slide_id', 'true_label', 'max_attr_LUAD', 'max_attr_LUSC']
    for col in df.columns:
        if col not in cols_to_keep and col != 'case_id':
            if df[col].sum() >= 5: 
                cols_to_keep.append(col)
                
    final_df = df[cols_to_keep]
    final_df.to_csv(output_csv_path, index=False)
    
    extracted_features = [c for c in final_df.columns if c not in ['slide_id', 'true_label', 'max_attr_LUAD', 'max_attr_LUSC']]


if __name__ == "__main__":
    visual_csv = '/content/AttriMIL-LungCancer/evaluation_results/extracted_visual_scores.csv'
    reports_folder = '/content/AttriMIL-LungCancer/tcga_reports'
    out_csv = '/content/AttriMIL-LungCancer/evaluation_results/ultimate_alignment_data.csv'
    
    build_ultimate_semantic_matrix(visual_csv, reports_folder, out_csv)