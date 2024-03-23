import json

TAJIKI_IN_FILE = '../01_OCR Tajiki/organize_out.json'
PERSIAN_IN_FILE = 'shahname_fa.json'

OUTPUT_ALIGNED_FILE = './aligned_chapters.json'

if __name__ == '__main__':
    with open(TAJIKI_IN_FILE) as f:
        tajiki_input = json.load(f)
    
    with open(PERSIAN_IN_FILE) as f:
        persian_input = json.load(f)

    output = []

    for (tajik_key, tajik_verse), (persian_key, persian_verse) in zip(tajiki_input.items(), persian_input.items()):
        output.append({
            'tajik_key': tajik_key,
            'persian_key': persian_key,
            'tajik_verse': tajik_verse,
            'persian_verse': sum(persian_verse.values(), [])
        })
        
    with open(OUTPUT_ALIGNED_FILE, 'w') as f:
        json.dump(output, f,  ensure_ascii=False)