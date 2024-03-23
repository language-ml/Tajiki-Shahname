from .gru import GRUModelForConditionalGeneration
from .transformer import TransformerForConditionalGeneration

def get_model(tokenizer, config_dict):
    kind = config_dict.pop('kind')
    
    model_constructor = {
        'gru': GRUModelForConditionalGeneration,
        'transformer': TransformerForConditionalGeneration
    }[kind]
    
    return model_constructor(
        vocab_size=tokenizer.vocab_size,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **config_dict
    )
