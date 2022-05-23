# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:37:47 2022

@author: Nasir
"""
import pandas as pd 
import numpy as np
dF=pd.read_csv(r'C:\Users\Nasir\Desktop\Data_Science\New folder\all_resumes.csv')

dF['Name'].str.contains('React', regex=True)

dF.info()

category = ['React','Java','Dotnet','SQL','Musquare']


dF["category"]=''
skils=["React","Peoplesoft","sql","workday"]

for i in dF[dF['Name'].str.contains(r'React')].values: 
    dF.loc[i[0],'category']='React'
    
for i in dF[dF['Name'].str.contains(r'SQL')].values: 
    dF.loc[i[0],'category']='sql'

for i in dF[dF['Name'].str.contains(r'workday')].values: 
    dF.loc[i[0],'category']='workday'
    
for i in dF[dF['Name'].str.contains(r'Peoplesoft')].values: 
    dF.loc[i[0],'category']='Peoplesoft'
    
df=dF
        
        
