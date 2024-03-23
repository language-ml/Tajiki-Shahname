import editdistance
import numpy as np

import wandb

def compute_metrics(y_true, y_pred):
    ed = [editdistance.eval(str1, str2) for str1, str2 in zip(y_true, y_pred)]
    data = list(zip(y_true, y_pred))[:8]
    return {
        'ed': np.mean(ed), 
        'samp': wandb.Table(data=data, columns=["True", "Pred"])
    }