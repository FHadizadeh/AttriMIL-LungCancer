from dataloader import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

import os
import torch
from torch import nn
import torch.optim as optim
import pdb
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from models.AttriMIL import AttriMIL
from utils import *

class Accuracy_Logger(object):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        return acc, correct, count

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.cuda(), label.cuda()
        slide_id = slide_ids.iloc[batch_idx]
        
        with torch.inference_mode():
            # FIXED: Added *rest to handle the extra Vision-Language variables safely
            logits, Y_prob, Y_hat, *rest = model(data)
            
        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
    else: 
        if n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
            auc_score = calc_auc(fpr, tpr)

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # FIXED: Updated all paths to map to Colab and Google Drive
    save_dir = '/content/drive/MyDrive/AttriMIL_Test_Results/'
    csv_path = './dataset_csv/test_data.csv'
    data_dir = '/content/extracted_features/'
    weight_dir = '/content/drive/MyDrive/AttriMIL_Weights/'
    
    # FIXED: Added the TCGA labels specific to your project
    label_dict = {'TCGA-LUAD':0, 'TCGA-LUSC':1}
    
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                                  data_dir = data_dir,
                                  shuffle = False, 
                                  print_info = True,
                                  label_dict = label_dict,
                                  patient_strat=False,
                                  ignore=[])
                                  
    os.makedirs(save_dir, exist_ok=True)
    model = AttriMIL(dim=512, n_classes=2).cuda()
    
    folds = [0]
    # FIXED: Re-routed to the specific "0" folder seen in your Google Drive image
    ckpt_paths = [os.path.join(weight_dir, str(fold), 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    
    all_results = []
    all_auc = []
    all_acc = []
    
    for ckpt_idx in range(len(ckpt_paths)):
        loader = get_simple_loader(dataset)
        model.load_state_dict(torch.load(ckpt_paths[ckpt_idx]))
        patient_results, test_error, auc, df, acc_logger = summary(model, loader, n_classes=2)
        
        # FIXED: Prevented the infinite nested array bug
        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
    
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(save_dir, save_name))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Test Complete: Evaluated in {elapsed_time:.2f} seconds.")
    print(f"Results saved to: {save_dir}")
