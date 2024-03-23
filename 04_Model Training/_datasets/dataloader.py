import torch
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding


def generate_dataloader(tokenizer, ds_train, ds_valid, train_bs, valid_bs):
    col_fn = DataCollatorForSeq2Seq(
        tokenizer, return_tensors='pt', padding='longest'
    )
    
    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=train_bs,
        collate_fn=col_fn,
        shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=valid_bs,
        collate_fn=col_fn,
        # shuffle=True
    )
    
    return train_loader, valid_loader

def generate_output_preprocess(tokenizer):
    def preprocess(all_input_ids):
        return_value = []
        for input_ids in all_input_ids:
            if tokenizer.eos_token_id in input_ids:
                input_ids = input_ids[:input_ids.index(tokenizer.eos_token_id)]
            return_value.append(tokenizer.decode(input_ids, skip_special_tokens=True))
        return return_value
    return preprocess
