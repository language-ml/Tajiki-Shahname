import json
import re

from sklearn.model_selection import train_test_split

INPUT_FILE_NAME = '../02_Align Verses/aligned_verse_and_chapters.json'
OUTPUT_FILE_NAME = 'parallel_corpus_splited.json'

def iranian_preprocess(text):
    persian_yeh = 'ی'   # ARABIC LETTER FARSI YEH U+06CC
    persian_kaf = 'ک'   # ARABIC LETTER KEHEH     U+06A9
    persian_heh = 'ه'   # ARABIC LETTER HEH       U+0647
    persian_alef = 'ا'  # ARABIC LETTER ALEF      U+0627
    persian_vav = 'و'   # ARABIC LETTER WAW       U+0648

    normalize_replace_map = {
        'ي': persian_yeh,   # ARABIC LETTER YEH                   U+064A
        'ى': persian_yeh,   # ARABIC LETTER ALEF MAKSURA          U+0649
        'ئ': persian_yeh,   # ARABIC LETTER YEH WITH HAMZA ABOVE  U+0626
        'ك': persian_kaf,   # ARABIC LETTER KAF                   U+0643
        'ة': persian_heh,   # ARABIC LETTER TEH MARBUTA           U+0629
        'أ': persian_alef,  # ARABIC LETTER ALEF WITH HAMZA ABOVE U+0623
        'إ': persian_alef,  # ARABIC LETTER ALEF WITH HAMZA BELOW U+0625
        'ؤ': persian_vav,   # ARABIC LETTER WAW WITH HAMZA ABOVE  U+0624
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
    ]

    trans = str.maketrans(
        normalize_replace_map | {char: None for char in normalize_remove_list}
    )
    return re.sub(r'[^زسیظدتمکعلصقژچاحغطگش هآب‌خپرنثجوذضف]', '', text.translate(trans))

def tajik_preprocess(text):
    return re.sub(r'[^иаюфхндкҷҳёгмжэръелшочзувйғқбпсӯтӣя \-]', '', text.lower())

if __name__ == '__main__':
    with open(INPUT_FILE_NAME, 'r') as f:
        input_file = json.load(f)

    all_tajik_verse = [
        tajik_preprocess(verse)
        for item in input_file
        for verse in item['tajik_verse']
    ]
    all_iranian_verse = [
        iranian_preprocess(verse)
        for item in input_file
        for verse in item['perian_verse']
    ]

    random_seed = 42

    tajik_train, tajik_valid_test, persian_train, persian_valid_test = train_test_split(
        all_tajik_verse,
        all_iranian_verse,
        test_size=0.2,
        random_state=random_seed
    )

    tajik_valid, tajik_test, persian_valid, persian_test = train_test_split(
        tajik_valid_test,
        persian_valid_test,
        test_size=0.5,
        random_state=random_seed
    )

    dataset = {
        "train": {
            "tajik": tajik_train,
            "persian": persian_train
        },
        "valid": {
            "tajik": tajik_valid,
            "persian": persian_valid
        },
        "test": {
            "tajik": tajik_test,
            "persian": persian_test
        }
    }

    with open(OUTPUT_FILE_NAME, 'w') as f:
        json.dump(dataset, f,  ensure_ascii=False)