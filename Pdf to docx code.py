# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:01:24 2022

@author: Nasir
"""
from pdf2docx import Converter
from glob import glob
import re
from win32com.client import constants
import os

paths = glob(r'C:\Users\Nasir\Desktop\Data_Science\New folder\Resumes\\**\\*.pdf', recursive=True)
pdf_file  =paths # source file 
for path in paths:
    size = len(path)
    docx_file = path[:size - 4] +'.docx'  # destination file

    def save_as_docx(path):
    #    Opening MS Word
        cv = Converter(path)
        cv.convert(docx_file, start=0, end=None)
        cv.close()
    save_as_docx(path)