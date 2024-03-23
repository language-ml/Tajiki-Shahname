import json
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

TEMPLATE_FOOTNOTE = np.load('footnote_template.npy')  # Used for cv2 template matching
INPUT_FILE_NAMES = {
    'shohnoma_1.pdf': slice(22, 475),
    'shohnoma_2.pdf': slice(4, 475),
    'shohnoma_3.pdf': slice(4, 473),
    'shohnoma_4.pdf': slice(4, 476),
    'shohnoma_5.pdf': slice(4, 476),
    'shohnoma_6.pdf': slice(4, 476),
    'shohnoma_7.pdf': slice(4, 473),
    'shohnoma_8.pdf': slice(4, 474),
    'shohnoma_9.pdf': slice(4, 476),
    'shohnoma_10.pdf': slice(4, 409)
}
OCR_WORKER_COUNT = 5
OUTPUT_FILENAME = 'ocr_out.json'

def filter_line(line):
    if len(line.strip()) > 3 and line.isupper():
        return 'Title'
    elif len(line.strip()) > 15:
        return 'Verse'
    else:
        return 'Remove'

def filter_text(text):
    return_value = []
    for line in text.strip('\n').split('\n'):
        filter_res = filter_line(line)
        if filter_res != 'Remove':
            return_value.append((filter_res, line))
    return return_value
    
def remove_footnote(image, template, th=1000):
    out = cv2.matchTemplate(np.asarray(image), template, cv2.TM_SQDIFF)
    # print(out.min())
    matched = out < th
    if out.min() < th:
        loc, _ = np.where(out == out.min())
        return Image.fromarray(np.asarray(image)[:loc[0]])
    else:
        return image
    
def do_ocr(image):
    return pytesseract.image_to_string(image, lang='tgk')

def job_fn(args):
    file_name, pages_slice = args
    return_value = []
    file_path = str(Path('PDF Files') / file_name)
    pdf_pages = convert_from_path(file_path)[pages_slice]

    for page_image in pdf_pages:
        page_image = remove_footnote(page_image, TEMPLATE_FOOTNOTE)
        return_value += filter_text(do_ocr(page_image))
        
    return return_value

if __name__ == '__main__':
    
    with multiprocessing.Pool(processes=OCR_WORKER_COUNT) as pool:
        all_books_results = pool.map(job_fn, INPUT_FILE_NAMES.items())

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(all_books_results, f, ensure_ascii=False)