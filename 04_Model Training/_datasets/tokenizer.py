import tokenizers
from transformers import PreTrainedTokenizerFast


def get_tokenizer():
    SPACE = ' '
    NIM_SPACE = '\u200c'
    PERSIAN_CHARS = 'آابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی'
    TAJIC_CHARS = '-абвгдежзийклмнопрстуфхчшъэюяёғқҳҷӣӯ'
    
    SOS_TOKEN = 'S'
    EOS_TOKEN = 'E'
    PAD_TOKEN = 'P'
    
    special_tokens = [EOS_TOKEN, PAD_TOKEN, SOS_TOKEN]
    all_chars = ''.join(special_tokens) + SPACE + NIM_SPACE + PERSIAN_CHARS + TAJIC_CHARS
    vocab = {c: idx for (idx, c) in enumerate(all_chars)}

    bpe_model = tokenizers.models.BPE(vocab=vocab, merges=[])
    tokenizer = tokenizers.Tokenizer(bpe_model)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.enable_padding(pad_token=PAD_TOKEN)
    # tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
    #     single=f"$0 {EOS_TOKEN}",
    #     pair=f"$A $B:0 {EOS_TOKEN}",
    #     special_tokens=[(EOS_TOKEN, vocab[EOS_TOKEN])],
    # )
    tokenizer.decoder = tokenizers.decoders.BPEDecoder()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_input_names=['input_ids', 'attention_mask']
    )
