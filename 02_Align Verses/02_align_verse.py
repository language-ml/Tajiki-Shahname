import re
import json
from collections import defaultdict

import editdistance
import numpy as np
from tqdm import tqdm

INPUT_FILE_NAME = 'aligned_chapters.json'
OUTPUT_FILE_NAME = 'aligned_verse_and_chapters.json'

TAJIK_INVALID_PATTERN = re.compile(r'[^иаюфхндкҷҳёгмжэръелшочзувйғқбпсӯтӣя \-]')
IRANIAN_INVALID_PATTERN = re.compile(r'[^زسیظدتمکعلصقژچاحغطگش هآب‌خپرنثجوذضف]')

NIM_SPACE = '\u200c'

FA_CONSONANT = {
    'تط': 't',
    'سثص': 's',
    'ق': 'v',
    'غ': 'q',
    'ضذزظ': 'z',
    'حه': 'h',
    'ب': 'b',
    'پ': 'p',
    'ج': 'j',
    'چ': 'c',
    'خ': 'x',
    'د': 'd',
    'ر': 'r',
    'ژ': 'i',
    'ش': 'u',
    'ع': 'a',
    'ف': 'f',
    'ک': 'k',
    'گ': 'g',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'آاوی': None
}

TJ_CONSONANT = {
    'т': 't',
    'с': 's',
    'қ': 'v',
    'ғ': 'q',
    'з': 'z',
    'ҳ': 'h',
    'б': 'b',
    'п': 'p',
    'ҷ': 'j',
    'ч': 'c',
    'х': 'x',
    'д': 'd',
    'р': 'r',
    'ж': 'i',
    'ш': 'u',
    'ъ': 'a',
    'ф': 'f',
    'к': 'k',
    'г': 'g',
    'л': 'l',
    'м': 'm',
    'н': 'n',
    'овяюёэеаӣйиуўӯ-': None
}

def make_trans(_input_dict):
    return_value = {}
    for key, val in _input_dict.items():
        for char in key:
            return_value[char] = val
    return str.maketrans(return_value)

def tajiki_consonant_extractor(text: str):
    trans_dict = make_trans(TJ_CONSONANT)
    return text.lower().translate(trans_dict)

def iranian_consonant_extractor(text:str):
    trans_dict = make_trans(FA_CONSONANT)
    
    H = 'ه'

    text = re.sub(fr'{H}(\s|{NIM_SPACE})', r'\1', text)
    return text.replace('ه ', ' ').translate(trans_dict)

def tajiki_preprocess(text: str) -> str:
    return_value = text.lower().strip('-\n\t ')
    return TAJIK_INVALID_PATTERN.sub('', return_value)

def iranian_preprocess(text: str) -> str:
    text = text.strip(f'-\n\t {NIM_SPACE}')
    text = re.sub(f'{NIM_SPACE}+', NIM_SPACE, text)
    text = re.sub(' +', ' ', text)
    return IRANIAN_INVALID_PATTERN.sub('', text)

def pair_verse(verse):
    return [(i // 2, verse[i], verse[i + 1]) for i in range(0, len(verse), 2)]

def tajiki_pipeline(text):
    return tajiki_consonant_extractor(tajiki_preprocess(text)).replace(' ', '')

def iranian_pipeline(text):
    return iranian_consonant_extractor(iranian_preprocess(text)).replace(' ', '').replace(NIM_SPACE, '')

def create_samet_space(_input, do_fn):
    return_value = defaultdict(list)
    for idx, m1, m2 in _input:
        return_value[(do_fn(m1), do_fn(m2))].append((idx, m1, m2))
    return return_value

def calc_distance(x1, x2):
    dist1 = editdistance.eval(x1[0], x2[0])
    dist2 = editdistance.eval(x1[1], x2[1])
    return dist1, dist2

def match_keys(tg_samet_keys, fa_samet_keys, th=3):
    exact_matches = set(tg_samet_keys) & set(fa_samet_keys)
    
    tg_nonmatched = set(tg_samet_keys) - exact_matches
    fa_nonmatched = set(fa_samet_keys) - exact_matches
        
    candidates = defaultdict(list)
    for tg_key in tg_nonmatched:
        for fa_key in fa_nonmatched:
            dist1, dist2 = calc_distance(tg_key, fa_key)
            if dist1 < th and dist2 < th:
                candidates[tg_key].append(fa_key)
    
    return exact_matches, candidates

def _orcale(tg_verse, *fa_verse):
    # A dirty fix for some edge cases!!!
    oracle_map = {
        1765: 1667,
        469: 339,
        777: 794,
        984: 994,
        1484: 994,
        2346: 2313,
        3033: 3003,
        20: 20,
        2520: 2484,
        3381: 3330,
        4051: 4011,
        1506: 1477
    }
    
    for item in fa_verse:
        if item[0] == oracle_map[tg_verse[0]]:
            return item
    raise Exception()
    
def nearest_match(tg_verses, fa_verses):
    if len(tg_verses) == len(fa_verses) == 1:
        return [(tg_verses[0], fa_verses[0])]
    else:
        return_value = []
        for tg_item in tg_verses:
            nearest = fa_verses[0]
            for fa_item in fa_verses:
                if abs(tg_item[0] - fa_item[0]) < abs(tg_item[0] - nearest[0]):
                    nearest = fa_item
            return_value.append((tg_item, nearest))
        return return_value

def match_and_pair(content_item):
    return_value = []
    
    tg_verse = pair_verse(content_item['tajik_verse'])
    fa_verse = pair_verse(content_item['persian_verse'])
    
    tg_verse_samet = create_samet_space(tg_verse, tajiki_pipeline)
    fa_verse_samet = create_samet_space(fa_verse, iranian_pipeline)
    
    exact_matches, candidates = match_keys(tg_verse_samet, fa_verse_samet)
    
    for key in exact_matches:
        return_value += nearest_match(tg_verse_samet[key], fa_verse_samet[key])
    
    for key, val in candidates.items():
        if len(val) == 1:
            return_value += nearest_match(tg_verse_samet[key], fa_verse_samet[val[0]])
        else:
            for tg_item in tg_verse_samet[key]:
                out = _orcale(tg_item, *(fa_verse_samet[fa_item][0] for fa_item in val))
                return_value.append((tg_item, out))
                
    return return_value

def remove_number(text):
    return re.sub(r'\d+', '', text).strip()

if __name__ == '__main__':
    with open(INPUT_FILE_NAME) as f:
        input_content = json.load(f)

    aligned_verse_result = []
    for book_chapter in tqdm(input_content):
        res = match_and_pair(book_chapter)
        aligned_verse_result.append(res)

    output_result = []

    for content_row, res in zip(input_content, aligned_verse_result):
        res = sorted(res, key=lambda x: x[0][0])  # by tajik verse number

        tajic_verse = []
        perian_verse = []

        for (_, tajic_v1, tajic_v2), (_, persian_v1, persian_v2) in res:
            tajic_verse.append(remove_number(tajic_v1))
            tajic_verse.append(remove_number(tajic_v2))
            perian_verse.append(persian_v1)
            perian_verse.append(persian_v2)

        assert len(tajic_verse) == len(perian_verse)

        output_result.append({
            'tajik_key': content_row['tajik_key'],
            'persian_key': content_row['persian_key'],
            'tajik_verse': tajic_verse,
            'perian_verse': perian_verse
        })

    with open(OUTPUT_FILE_NAME, 'w') as f:
        json.dump(output_result, f, ensure_ascii=False)