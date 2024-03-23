# !pip install pypdf pytesseract pdf2image tesserocr
import collections
from pathlib import Path

import pytesseract
from pypdf import PdfReader
from pdf2image import convert_from_path


class EzPDF:
    def __init__(self, file_name, pages, images):
        self.file_name = file_name
        
        assert len(pages) == len(images)
        self.pages = pages
        self.images = images
    
    @property
    def page(self):
        assert len(self) == 1
        return self.pages[0]
    
    @property
    def image(self):
        assert len(self) == 1
        return self.images[0]
        
    @classmethod
    def load_file(cls, file_name):
        if not Path(file_name).exists():
            raise ValueError()
        return cls(
            file_name=file_name,
            pages=list(PdfReader(file_name).pages),
            images=convert_from_path(file_name)
        )
            
    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self):
                raise IndexError()
            index = slice(index, index + 1)
            
        return self.__class__(
            file_name=self.file_name,
            pages=self.pages[index],
            images=self.images[index],
        )
    
    def __repr__(self):
        return f"{self.file_name} len: {len(self)}"
    

def couple_text(**kwargs):
    page_info = collections.namedtuple('Page', ['number', *kwargs.keys()])
    return [
        page_info(idx + 1, *[text.split('\n') for text in page_contents])
        for idx, page_contents
        in enumerate(zip(*kwargs.values()))
    ]

def _auto_process_list(func):
    def wrapper(_input):
        if isinstance(_input, list):
            return [func(item) for item in _input]
        return func(_input)
    return wrapper

@_auto_process_list
def extract_text(page):
    return page.extract_text()

@_auto_process_list
def do_ocr(image):
    return pytesseract.image_to_string(image, lang='tgk')
