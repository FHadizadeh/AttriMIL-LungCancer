import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

def plot_ultimate_clinical_heatmap(csv_path, output_path):
    df = pd.read_csv(csv_path)

    df['max_attr_LUAD'] = (df['max_attr_LUAD'] - df['max_attr_LUAD'].min()) / (df['max_attr_LUAD'].max() - df['max_attr_LUAD'].min())
    df['max_attr_LUSC'] = (df['max_attr_LUSC'] - df['max_attr_LUSC'].min()) / (df['max_attr_LUSC'].max() - df['max_attr_LUSC'].min())
    
    fixed_cols = ['slide_id', 'true_label', 'max_attr_LUAD', 'max_attr_LUSC']
    clinical_features = [col for col in df.columns if col not in fixed_cols]
    visual_branches = ['max_attr_LUAD', 'max_attr_LUSC']
    
    corr_matrix = np.zeros((len(clinical_features), len(visual_branches)))
    p_matrix = np.zeros((len(clinical_features), len(visual_branches)))

    for i, feature in enumerate(clinical_features):
        for j, branch in enumerate(visual_branches):
            binary_data = df[feature].values
            continuous_data = df[branch].values
            
            if len(np.unique(binary_data)) > 1:
                corr, p_val = pointbiserialr(binary_data, continuous_data)
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
            else:
                corr_matrix[i, j] = 0
                p_matrix[i, j] = 1.0

    labels = []
    for i in range(len(clinical_features)):
        row_labels = []
        for j in range(len(visual_branches)):
            val = corr_matrix[i, j]
            p = p_matrix[i, j]
            star = ""
            if p < 0.001: star = "***"
            elif p < 0.01: star = "**"
            elif p < 0.05: star = "*"
            row_labels.append(f"{val:.2f}{star}")
        labels.append(row_labels)

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    
    ax = sns.heatmap(
        corr_matrix, 
        annot=np.array(labels), 
        fmt="", 
        cmap='RdBu_r', 
        center=0,
        vmin=-0.5, 
        vmax=0.5,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Point-Biserial Correlation ($r_{pb}$)"}
    )

    ax.set_yticklabels(clinical_features, rotation=0, fontsize=10)
    ax.set_xticklabels(['LUAD Visual Score', 'LUSC Visual Score'], fontsize=12, fontweight='bold')
    
    plt.title('Clinical Validation: Visual-Language Alignment\n(Stars indicate statistical significance: *p<0.05, **p<0.01, ***p<0.001)', 
              fontsize=14, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_ultimate_clinical_heatmap(
        '/content/AttriMIL-LungCancer/evaluation_results/ultimate_alignment_data.csv',
        '/content/AttriMIL-LungCancer/evaluation_results/FINAL_INTERPRETABILITY_PLOT.png'
    )