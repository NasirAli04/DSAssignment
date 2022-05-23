# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:35:29 2022

@author: Nasir
"""
import os
import docx2txt
import glob
import pandas as pd       

text = ''
for file in glob.glob('Data\*.docx'):
    text += docx2txt.process(file)
    
    try:
        file.remove('word/document.xml')

    except: 
        pass
    

os.chdir(r'C:\Users\Nasir\Desktop\Data_Science\New folder')

Descriptions = []
name = []

for file in glob.glob('**\\*.docx'):
    Descriptions.append(docx2txt.process(file))    
    name.append(file)

data = pd.DataFrame(
    {'Descriptions': Descriptions,
     'Name': name,
    })

for file in glob.glob('**\\**\\*.docx'):
    Descriptions.append(docx2txt.process(file))    
    name.append(file)

data1 = pd.DataFrame(
    {'Descriptions': Descriptions,
     'Name': name,
    })
data.dtypes
df=data1

df.to_csv('all_resumes.csv')
