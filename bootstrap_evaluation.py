import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def calculate_bootstrap_metrics(csv_path, n_iterations=1000):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return

    y_true_all = df['Y'].values
    y_pred_all = df['Y_hat'].values
    y_prob_all = df['p_1'].values # احتمال کلاس ۱ برای محاسبه AUC
    
    bootstrapped_auc = []
    bootstrapped_acc = []
    bootstrapped_f1 = []
    bootstrapped_prec = []
    bootstrapped_rec = []

    n_samples = len(y_true_all)

    np.random.seed(42)

    for i in tqdm(range(n_iterations)):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        
        y_true_boot = y_true_all[indices]
        y_prob_boot = y_prob_all[indices]
        y_pred_boot = y_pred_all[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        bootstrapped_auc.append(roc_auc_score(y_true_boot, y_prob_boot))
        bootstrapped_acc.append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrapped_f1.append(f1_score(y_true_boot, y_pred_boot, average='weighted'))
        bootstrapped_prec.append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        bootstrapped_rec.append(recall_score(y_true_boot, y_pred_boot))

    print("-" * 50)
    print(f"AUC-ROC:   {np.mean(bootstrapped_auc):.4f} ± {np.std(bootstrapped_auc):.4f}")
    print(f"Accuracy:  {np.mean(bootstrapped_acc):.4f} ± {np.std(bootstrapped_acc):.4f}")
    print(f"F1-Score:  {np.mean(bootstrapped_f1):.4f} ± {np.std(bootstrapped_f1):.4f}")
    print(f"Precision: {np.mean(bootstrapped_prec):.4f} ± {np.std(bootstrapped_prec):.4f}")
    print(f"Recall:    {np.mean(bootstrapped_rec):.4f} ± {np.std(bootstrapped_rec):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    calculate_bootstrap_metrics(csv_path='/content/AttriMIL-LungCancer/evaluation_results/fold_0.csv', n_iterations=1000)