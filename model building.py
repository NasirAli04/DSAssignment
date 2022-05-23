# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:32:59 2022

@author: Nasir
"""


import pandas as pd 
import numpy as np
import seaborn as sns 
from sklearn import preprocessing
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,RepeatedKFold,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
    

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Descriptions'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

le = preprocessing.LabelEncoder()
df['category'] = le.fit_transform(df.category.values)
y=df['category']
# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

print("Naive bayes accuracy")
print(accuracy*100)

#selecting best models
model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators = 10),
             GradientBoostingRegressor()]

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state= None)
cv_results = []
cv_results_mean =[]
for ele in model_selc:
    cross_results = cross_val_score(ele, X_train, y_train, cv=kfold, scoring ='r2')
   
    cv_results.append(cross_results)
   
    cv_results_mean.append(cross_results.mean())
    print("\n MODEL: ",ele,"\nMEAN R2:",cross_results.mean())



regressor = RandomForestRegressor(n_estimators =50, random_state = 0)
# fit the regressor with x and y data
regressor.fit(X_train, y_train) 
# Use the forest's predict method on the test data
y_pred= regressor.predict(X_test)
y_pred
# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
r2_score = regressor.score(X_test,y_test)
print(r2_score*100,'%')
RandomForestRegressor_Accuracy=r2_score*100




