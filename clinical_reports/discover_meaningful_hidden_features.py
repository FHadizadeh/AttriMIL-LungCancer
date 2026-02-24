import os
import re
import PyPDF2
from tqdm import tqdm
from collections import defaultdict

def discover_meaningful_hidden_features(reports_dir):
    
    TARGET_FEATURES = {
        'glandular', 'glands', 'acinar', 'acini', 'adenocarcinoma', 'aca', 'nsclc',
        'squamous', 'sqcc', 'scc', 'keratinization', 'keratin', 'mucin', 'mucinous',
        'papillary', 'lepidic', 'necrosis', 'necrotic', 'basaloid', 'pleomorphism', 
        'pleomorphic', 'mitosis', 'mitotic', 'differentiation', 'differentiated', 
        'invasion', 'invasive', 'vascular', 'lvi'
    }

    ADMIN_WORDS = {
        'report', 'patient', 'diagnosis', 'specimen', 'size', 'cm', 'mm', 'block', 
        'slide', 'material', 'right', 'left', 'upper', 'lower', 'lobe', 'lung', 
        'tissue', 'tumor', 'carcinoma', 'cell', 'show', 'seen', 'bronchial', 
        'frozen', 'section', 'identified', 'labeled', 'level', 'lymph', 'node', 
        'nodes', 'margin', 'measuring', 'negative', 'pleura', 'pleural', 'received', 
        'resection', 'submitted', 'tan', 'gross', 'description', 'microscopic', 
        'clinical', 'history', 'cassette', 'formalin', 'fixed', 'stain', 'positive', 
        'present', 'absent', 'foci', 'focus', 'involved', 'uninvolved', 'type', 
        'grade', 'stage', 'margins', 'blocks', 'sections', 'carcinomas', 'tumors', 
        'cells', 'this', 'there', 'mass', 'fresh', 'consists', 'black', 'surface', 
        'entirely', 'biopsy', 'dimension', 'soft', 'parenchyma', 'part', 'malignancy',
        'excision', 'greatest', 'hilar', 'pathology', 'x', 'p', 't', 'n', 'm'
    }

    doc_frequency = defaultdict(int)
    total_docs = 0

    for filename in tqdm(os.listdir(reports_dir)):
        if filename.upper().endswith(".PDF"):
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
                total_docs += 1
                clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
                words = clean_text.split()
                
                valid_words = [w for w in words if len(w) > 3 and w not in ADMIN_WORDS and w not in TARGET_FEATURES]
                
                unique_words_in_doc = set(valid_words)
                for w in unique_words_in_doc:
                    doc_frequency[w] += 1

    meaningful_words = {word: count for word, count in doc_frequency.items() if 15 <= count <= 180}
    
    sorted_words = sorted(meaningful_words.items(), key=lambda item: item[1], reverse=True)

    print(f"\n\n💎 ۲۰ کلمه‌ی مهمِ جا مانده (با فیلتر فرکانسِ طلایی بین 15 تا 180 از {total_docs} گزارش):")
    for word, count in sorted_words[:20]:
        print(f" - {word}: در {count} گزارش حضور داشت")

if __name__ == "__main__":
    reports_folder = '/content/AttriMIL-LungCancer/tcga_reports'
    discover_meaningful_hidden_features(reports_folder)