import numpy as np
import torch
from tqdm import tqdm

from _utils import prefix_dict_keys

def train_loop(model, loader, optimizer, gradient_clipping=None, use_tqdm=False):
    model.train()

    batch_losses = []
    
    if use_tqdm:
        loader = tqdm(loader, position=2, desc="Train Loop", leave=False)
        
    for row in loader:
        optimizer.zero_grad()
 
        out = model(**row.to(model.device))
        loss = out.loss
 
        batch_loss_value = loss.item()
        loss.backward()
        
        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        
        optimizer.step()
 
        batch_losses.append(batch_loss_value)
    
    loss_value = np.mean(batch_losses)
    return prefix_dict_keys('train', {
        'loss': loss_value
    })

def _predict(model, row):
    return model.generate(
        input_ids=row.input_ids,
        attention_mask=row.attention_mask,
        max_length=50
    )


def valid_loop(model, loader, compute_metrics, output_preprocess, use_tqdm=False):
    model.eval()
    
    all_true = []
    all_pred = []
    
    if use_tqdm:
        loader = tqdm(loader, position=2, desc="Valid Loop", leave=False)
        
    with torch.no_grad():
        for row in loader:
            row.to(model.device)
            pred = _predict(model, row)
            
            all_true += row.labels.detach().cpu().tolist()
            all_pred += pred.detach().cpu().tolist()
                                    
    all_true = output_preprocess(all_true)
    all_pred = output_preprocess(all_pred)
        
    metrics = compute_metrics(y_true=all_true, y_pred=all_pred)
    return_value = {
        **metrics
    }
    
    return prefix_dict_keys('valid', return_value)