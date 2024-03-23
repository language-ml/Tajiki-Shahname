import datasets
import json

def _load_dataset(dict_content):
    return datasets.Dataset.from_dict({
        "tajik": dict_content["tajik"],
        "persian": dict_content["persian"]
    })

def load_dataset(path):
    with open(path) as f:
        content = json.load(f)
    
    return _load_dataset(content)

def load_dict_dataset(path):
    with open(path) as f:
        content = json.load(f)

    all_datasets = {
        split: _load_dataset(content[split]) for split in content.keys()
    }
    
    return datasets.DatasetDict(all_datasets)

def concat_all(str_list, prefix='', postfix=''):
    return [prefix + item + postfix for item in str_list]

def map_dataset(dataset, tokenizer, source_key, target_key):
    def map_function(input_dict):
        source = concat_all(input_dict[source_key], prefix=tokenizer.bos_token, postfix=tokenizer.eos_token)
        target = concat_all(input_dict[target_key], postfix=tokenizer.eos_token)
        return {
            **tokenizer(source),
            'labels': tokenizer(target).input_ids
        }
    
    dataset = dataset.map(map_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset