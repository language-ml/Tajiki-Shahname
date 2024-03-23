import json
import itertools
from collections import defaultdict

INPUT_FILE_NAME = 'ocr_out.json'
OUTPUT_FILE_NAME = 'organize_out.json'

TAJIK_CHAPTERS = [
    'ОҒОЗИ КИТОБ',
    'ОҒОЗИ ДОСТОН',
    'ҲУШАНГ',
    'ТАҲМУРАС',
    'ҶАМШЕД',
    'ЗАҲҲОК',
    'ФАРИДУН',
    'МАНУЧЕҲР',
    'НӮЗАР',
    'ЗАВИ ТАҲМОСП',
    'ГАРШОСП',
    'КАЙҚУБОД',
    'КАЙКОВУС',
    'БАРБАРИСТОН ВА ДИГАР ДОСТОНҲО',
    'СӮҲРОБ',
    'ДОСТОНИ СИЁВУШ',
    'КАЙХУСРАВ',
    'ОҒОЗИ ДОСТОНИ ФУРУД - ПИСАРИ СИЁВУШ',
    'ДОСТОНИ КОМУСИ КАШОНӢ',
    'ДОСТОНИ РУСТАМ БО ХОҚОНИ ЧИН',
    'ДОСТОНИ ҶАНГИ РУСТАМ БО АКВОНДЕВ',
    'ДОСТОНИ БЕЖАН БО МАНИЖА',
    'ДОСТОНИ ДУВОЗДАҲ РУХ',
    'ПОДШОҲИИ КАЙХУСРАВ',
    'ПОДШОҲИИ ЛӮҲРОСП',
    'ПОДШОҲИИ ГУШТОСП',
    'ДОСТОНИ ҲАФТ ХОН',
    'ДОСТОНИ РАЗМИ ИСФАНДЁР',
    'ДОСТОНИ РУСТАМ ВА ШАҒОД',
    'ПОДШОҲИИ БАҲМАНИ ИСФАНДЁР',
    'ПОДШОҲИИ ҲУМОЙ',
    'ПОДШОҲИИ ДОРОБ',
    'ПОДШОҲИИ ДОРО ПИСАРИ ДОРОБ',
    'ПОДШОҲИИ СИКАНДАР',
    'ПОДШОҲИИ АШКОНИЁН',
    'ПОДШОҲИИ АРДАШЕРИ БОБАКОН',
    'ПОДШОҲИИ ШОПУРИ АРДАШЕР',
    'ПОДШОҲИИ УРМУЗДИ ШОПУР',
    'ПОДШОҲИИ БАҲРОМИ УРМУЗД',
    'БАҲРОМ ВА АНДАРЗ КАРДАН БА',
    'БАҲРОМИЁН ВА МУРДАНАШ ПАС',
    'ПОДШОҲИИ НАРСИИ',
    'ПОДШОҲИИ УРМУЗДИ',
    'ПОДШОҲИИ ШОПУРИ',
    'ПОДШОҲИИ АРДАШЕРИ',
    'ИБНИ ШОПУР',
    'ПОДШОҲИИ БАҲРОМ ПИСАРИ',
    'ПОДШОҲИИ ЯЗДГИРДИ БАЗАГАР',
    'ПОДШОҲИИ БАҲРОМИ ГӮР',
    'ПОДШОҲИИ ЯЗДГИРД ПИСАРИ',
    # 'ПОДШОҲИИ ҲУРМУЗ',
    # 'ПОДШОҲИИ ПИРУЗ - ПИСАРИ'
    # 'ПОДШОҲИИ БАЛОШ',
    'ПОДШОҲИИ ҚУБОДИ ПИРӮЗ',
    'ПОДШОҲИИ КИСРОИ',
    'ПОДШОҲИИ ҲУРМУЗД',
    'ПОДШОҲИИ ХУСРАВИ ПАРВИЗ',
    'ПОДШОҲИИ ҚУБОДИ ПАРВИЗ',
    'АРДАШЕРИ ШИРӮЙ',
    'ПОДШОҲИИ ФАРОИНИ ГУРОЗ',
    'ПОДШОҲИИ ПӮРОНДУХТ',
    'ПОДШОҲИИ ОЗАРМДУХТ',
    'ПОДШОҲИИ ФАРРУХЗОД',
    'ПОДШОҲИИ ЯЗДГИРД'
]

EXCEPTION_LIST = [
    'Подшоҳии Каюмарс - аввалини мулуки Аҷам сӣ сол буд',
    'Подшоҳии Ҳушанг чиҳил сол буд',
    'Подшоҳии Таҳмураси девбанд сӣ сол буд',
    'Подшоҳии Ҷамшед ҳафтсад сол буд',
    'Подшоҳии Заҳҳок ҳазор сол буд',
    'Подшоҳии ӯ саду бист сол буд',
    'р: у рухсора зар,',
    'Подшоҳии ӯ панҷ сол буд',
    'Подшоҳии ӯ нӯҳ сол буд',
    'Подшоҳии ӯ сад сол буд',
    'Чунон” дону э ашав з-ӯ ба',
    'Як солу ду моҳ буд'
]

def group_poem(all_verse):
    create_poem = lambda: {"Title": [], 'Verse': []}
    return_value = [create_poem()]

    for kind, line in all_verse:
        if kind == 'Title':
            if len(return_value[-1]['Verse']) != 0:
                return_value.append(create_poem())
        return_value[-1][kind].append(line)
    return return_value

def organize_by_chapters(poems_in_groupes):
    return_value = defaultdict(list)

    chapter_idx = 0
    selected_chapter = None

    for group in all_books_groupes:
        if chapter_idx < len(TAJIK_CHAPTERS) and TAJIK_CHAPTERS[chapter_idx] in group['Title']:
            selected_chapter = TAJIK_CHAPTERS[chapter_idx]
            chapter_idx += 1
        if len(group['Verse']) > 8:
            new_verses = [verse for verse in group['Verse'] if verse not in EXCEPTION_LIST]
            return_value[selected_chapter] += new_verses
        
    return return_value

if __name__ == '__main__':
    with open(INPUT_FILE_NAME) as f:
        all_books_res = json.load(f)
        
    all_books_groupes = [
        item 
        for book_res in all_books_res 
        for item in group_poem(book_res)
    ]
    
    organized_poems = organize_by_chapters(all_books_groupes)
    
    with open(OUTPUT_FILE_NAME, 'w') as f:
        json.dump(organized_poems, f,  ensure_ascii=False)
