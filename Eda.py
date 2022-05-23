# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:37:47 2022

@author: Nasir
edits added by vennela
"""
import pandas as pd 
import numpy as np
import seaborn as sns 
from nltk.tokenize import RegexpTokenizer

dF=pd.read_csv(r'C:\Users\Nasir\Desktop\Data_Science\New folder\all_resumes.csv')

dF['Name'].str.contains('React', regex=True)

dF.info()



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

df['Descriptions'] = df['Descriptions'].astype(str).str.lower()
df.head(3)
df.info()
df.rename(columns = {'Unnamed: 0':'index'}, inplace = True)
df["index"] = df["index"].astype(str).astype(int)
  
#sns.barplot(x=df["category"], y=df["index"],data=df)
df['category'].value_counts()[:20].plot(kind='barh')

regexp = RegexpTokenizer('\w+')
df['text_token']=df['Descriptions'].apply(regexp.tokenize)
df.head(3)
import nltk
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")
# Extend the list with your own custom stopwords
my_stopwords = ['https']
stopwords.extend(my_stopwords)
# Remove stopwords
df1=df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
df.head(3)

df1=df['text_token'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
df.head(3)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()



df["React"]=''
for i in df[df['Name'].str.contains(r'React')].values: 
    df.loc[i[0],'React']= df.loc[i[0],'text_token']
df["Sql"]=''   
for i in df[df['Name'].str.contains(r'SQL')].values: 
    df.loc[i[0],'Sql']= df.loc[i[0],'text_token']
df["workday"]=''
for i in df[df['Name'].str.contains(r'workday')].values: 
    df.loc[i[0],'workday']= df.loc[i[0],'text_token']
df["Peoplesoft"]=''   
for i in df[df['Name'].str.contains(r'Peoplesoft')].values: 
    df.loc[i[0],'Peoplesoft']= df.loc[i[0],'text_token']
    
    

import matplotlib.pyplot as plt
from wordcloud import WordCloud
df['text_token'] = df['text_token'].apply(wordnet_lem.lemmatize)
all_words = ' '.join([word for word in df['text_token']])

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

df['React'] = df['React'].apply(wordnet_lem.lemmatize)
all_React_words = ' '.join([word for word in df['React']])

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_React_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

df['Sql'] = df['Sql'].apply(wordnet_lem.lemmatize)
all_Sql_words = ' '.join([word for word in df['Sql']])

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_Sql_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

df['workday'] = df['workday'].apply(wordnet_lem.lemmatize)
all_workday_words = ' '.join([word for word in df['workday']])

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_workday_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');

df['Peoplesoft'] = df['Peoplesoft'].apply(wordnet_lem.lemmatize)
all_Peoplesoft_words = ' '.join([word for word in df['Peoplesoft']])

wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_Peoplesoft_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');



from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

words = nltk.word_tokenize(all_words)
fd = FreqDist(words)
fd.most_common(3)   
# Obtain top 10 words
top_10 = fd.most_common(50)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))     
import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()  
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()



words = nltk.word_tokenize(all_React_words)
fd = FreqDist(words)
fd.most_common(3)   
# Obtain top 10 words
top_10 = fd.most_common(20)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))     
import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()  
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()

words = nltk.word_tokenize(all_Sql_words)
fd = FreqDist(words)
fd.most_common(3)   
# Obtain top 10 words
top_10 = fd.most_common(20)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))     
import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()  
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()
################################################
words = nltk.word_tokenize(all_workday_words)
fd = FreqDist(words)
fd.most_common(3)   
# Obtain top 10 words
top_10 = fd.most_common(20)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))     
import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()  
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()

words = nltk.word_tokenize(all_Peoplesoft_words)
fd = FreqDist(words)
fd.most_common(3)   
# Obtain top 10 words
top_10 = fd.most_common(20)

# Create pandas series to make plotting easier
fdist = pd.Series(dict(top_10))     
import seaborn as sns
sns.set_theme(style="ticks")

sns.barplot(y=fdist.index, x=fdist.values, color='blue');
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()  
import plotly.express as px

fig = px.bar(y=fdist.index, x=fdist.values)

# sort values
fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})

# show plot
fig.show()
