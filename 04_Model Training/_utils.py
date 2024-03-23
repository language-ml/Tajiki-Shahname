def prefix_dict_keys(prefix, input_dict):
    return {f'{prefix}_{key}': val for key, val in input_dict.items()}

def print_system_info():
    from platform import python_version
    print(f"Python version is: {python_version()}")
    
    try:
        import sklearn
        print(f"Scikit-learn version is: {sklearn.__version__}")
    except:
        print("Scikit-learn not found!!!")
    
    try:
        import torch
        print(f"Torch version is: {torch.__version__}")
        if torch.cuda.is_available() and torch.cuda.device_count() >= 0:
            print(f"Nvidia device is: {torch.cuda.get_device_name(0)}")
        else:
            print("Torch is using CPU")
    except:
        print("Torch not found!!!")
        return

    try:
        import transformers
        print(f"Transformers version is: {transformers.__version__}")
        try:
            print(f"Adapterhub version is: {transformers.adapters.__version__}")
        except:
            print("Adapterhub not found!!!")
    except:
        print("Transformers not found!!!")

def silent_logs():
    import os
    import warnings
    
    warnings.filterwarnings("ignore")
    os.environ["WANDB_SILENT"] = "true"
    # os.environ["TRANSFORMERS_VERBOSITY"] = "fatal"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    import transformers
    from transformers.utils import logging
    logging.set_verbosity(transformers.logging.FATAL)
    from datasets.utils.logging import disable_progress_bar, set_verbosity_error
    disable_progress_bar()
    set_verbosity_error()

def sp_encode(data):
    import json
    import base64
    return base64.b32encode(json.dumps(data).encode())

def sp_decode(encoded_data):
    import json
    import base64
    return json.loads(base64.b32decode(encoded_data).decode())



def persian_preprocess(text):
    persian_yeh = 'ی'   # ARABIC LETTER FARSI YEH U+06CC
    persian_kaf = 'ک'   # ARABIC LETTER KEHEH     U+06A9
    persian_heh = 'ه'   # ARABIC LETTER HEH       U+0647
    persian_alef = 'ا'  # ARABIC LETTER ALEF      U+0627
    persian_vav = 'و'   # ARABIC LETTER WAW       U+0648
    persian_lam = 'ل'   # ARABIC LETTER LAM       U+0644
    
    persian_allah = persian_lam + persian_lam + persian_heh

    normalize_replace_map = {
        'ي': persian_yeh,    # ARABIC LETTER YEH                   U+064A
        'ى': persian_yeh,    # ARABIC LETTER ALEF MAKSURA          U+0649
        'ئ': persian_yeh,    # ARABIC LETTER YEH WITH HAMZA ABOVE  U+0626
        'ك': persian_kaf,    # ARABIC LETTER KAF                   U+0643
        'ة': persian_heh,    # ARABIC LETTER TEH MARBUTA           U+0629
        'ۀ': persian_heh,    # ARABIC LETTER HEH WITH YEH ABOVE    U+06C0
        'أ': persian_alef,   # ARABIC LETTER ALEF WITH HAMZA ABOVE U+0623
        'إ': persian_alef,   # ARABIC LETTER ALEF WITH HAMZA BELOW U+0625
        'ؤ': persian_vav,    # ARABIC LETTER WAW WITH HAMZA ABOVE  U+0624
        'ﷲ': persian_allah,  # ARABIC LIGATURE ALLAH ISOLATED FORM U+FDF2
        '\xa0': ' '
    }

    normalize_remove_list = [
        'َ',   # ARABIC FATHA         U+064E
        'ُ',   # ARABIC DAMMA         U+064F
        'ِ',   # ARABIC KASRA         U+0650
        'ً',   # ARABIC FATHATAN      U+064B
        'ٌ',   # ARABIC DAMMATAN      U+064C
        'ٍ',   # ARABIC KASRATAN      U+064D
        'ّ',   # ARABIC SHADDA        U+0651
        'ْ',   # ARABIC SUKUN         U+0652
        'ٓ',   # ARABIC MADDAH ABOVE  U+0653
        'ٔ',   # ARABIC HAMZA ABOVE   U+0654
        'ٕ',   # ARABIC HAMZA BELOW   U+0655
        'ـ',  # ARABIC TATWEEL       U+0640
        'ء',  # ARABIC LETTER HAMZA  U+0621
        
    ]

    trans = str.maketrans(
        normalize_replace_map | {char: None for char in normalize_remove_list}
    )
    text = text.translate(trans)
    text = re.sub(r'[^زسیظدتمکعلصقژچاحغطگش هآب‌خپرنثجوذضف]', '', text)
    text = re.sub(' +', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.replace(' \u200c', '\u200c').replace('\u200c ', '\u200c')
    return text