#2/28/18
###analyzing 2016 presidential polls in north carolina 
nc=pd.read_csv("upshot-siena-polls.csv")
nc.info()
#subset for north carolina
s='North Carolina'
nc['state']
nc_state=[y for y in nc['state']if s in y]
nc['nc']=nc['state'].str.contains('North Carolina')
nc['nc']=nc['nc'].astype(int)
nc_state=nc[nc.nc==1]

#hot-encode democrat candidate
vote_dem={'Unfavorable':0,
'Favorable':1,"[DO NOT READ] Don't know/No opinion":2}
nc_state['dem_fav']=nc_state['d_pres_fav'].replace(vote_dem)
nc_state.head(4)
#republican favorite
vote_rep={'Unfavorable':0,
'Favorable':1,"[DO NOT READ] Don't know/No opinion":2}
nc_state['rep_fav']=nc_state['r_pres_fav'].replace(vote_rep)
nc_state.shape 
nc_state['vt_pres_2'].unique() 
#hot-encode vt_pres_2
prez_fav={'Donald Trump, the Republican':0,
'Hillary Clinton, the Democrat':1,
"DO NOT READ] Won't vote":2,
"[DO NOT READ] Don't know/No opinion":3,
"[DO NOT READ] Someone else (specify)":4,
"nan":5} 

nc_state['pres_fav']=nc_state['vt_pres_2'].replace(prez_fav)
nc_state['vt_sen'].unique() 
#senator 
sen_fav={'Richard Burr, the Republican':0,
"[DO NOT READ] Don't know/No opinion":1,
"[DO NOT READ] Won't vote":2,
"Deborah Ross, the Democrat":3,
"[DO NOT READ] Someone else (specify)":4
}
nc_state['sen_fav']=nc_state['vt_sen'].replace(sen_fav)
#governor
nc_state['vt_gov'].unique()
gov_fav={"Pat McCrory, the Republican":0,
"Roy Cooper, the Democrat":1,
"[DO NOT READ] Don't know/No opinion":2,
"[DO NOT READ] Won't vote":3,"[DO NOT READ] Someone else (specify)":4}

nc_state['gov_fav']=nc_state['vt_gov'].replace(gov_fav)

#scale (vote likelihood)
nc_state['scale'].unique() 
vote_lh={"Ten (definitely will vote)":10,
"Nine":9,"Three":3,"One (definitely will NOT vote)":1,
"Eight":8,"Five":5,"Two":2,"Six":6,'Seven':7,
"Four":4,"[DO NOT READ] Don't know/No opinion":5,"nan":5}
nc_state['vote_lh']=nc_state['scale'].replace(vote_lh)

#party id
nc_state['partyid'].unique() 
party_id={'Republican':0,
'Democrat':1,'Independent (No party)':2,
'[DO NOT READ] Refused':3,'or as a member of another political party':4} 
nc_state['party']=nc_state['partyid'].replace(party_id)

#gender
nc_state['gender'].unique() 
gender_x={'Male':0,
'Female':1}
nc_state['gender_x']=nc_state['gender'].replace(gender_x)

#education
nc_state['educ'].unique() 
education={'Graduate or Professional degree':1,
'Some college or trade school':1,"Bachelors' degree":1,
'High school':0,'[DO NOT READ] Refused':2,
'Grade school':1}
nc_state['education']=nc_state['educ'].replace(education)

#race
nc_state['race'].unique() 
race_card={'Caucasian/White':0,
'[DO NOT READ] Other/Something else (specify)':1,
'African American/Black':2,'[DO NOT READ] Refused':3,
'Asian':4}
nc_state['race_card']=nc_state['race'].replace(race_card)

#hispanic
nc_state['hisp'].unique() 
hispanic={'No':0,'Yes':1,'[DO NOT READ] Refused':2}
nc_state['hispanic']=nc_state['hisp'].replace(hispanic) #5.1% hispanic? 
nc_state['hispanic'].value_counts()  

####weighted vote likelihood
#i. clinton voters
nc_dem=nc_state[nc_state['pres_fav']==1] 
nc_dem.shape #720 #30.3% 
nc_dem['vote_lh']=nc_dem['vote_lh'].astype(float)
nc_dem.describe() #mean 9.43
#clinton vote score =(720/len(nc_state))*9.43 (2.86 weighted score)

#ii. trump vote
nc_rep=nc_state[nc_state['pres_fav']==0]
nc_rep.shape #685 28.9%
nc_rep.describe() #9.573 
#trump weighted vote score #2.76 
nc_state['pres_fav'].unique()

#background trump 49.8%, clinton 46.2 (+3.6 trump edge)
#i. vote count,vote lh (2.86 clinton,2.76 trump)**
#(+0.9 clinton edge)

#ii.adding candiate favorbility**
nc_dem['dem_fav'].value_counts() #632 (0.8951)
nc_rep['rep_fav'].value_counts() #548 (0.869)

#iii.  adding sentator and governor voting preference
nc_state['sen_fav'].unique() #0 rep, 3 dem 
nc_state['sen_fav'].value_counts() #rep=1044 (0.494),dem=1070 (0.506)
nc_state['gov_fav'].unique() #0 rep, 1 dem 
nc_state['gov_fav'].value_counts() #rep=1042 (0.473),dem=1162 (0.527)

#score update (i,ii, iii)
#clinton +4.788
#trump +4.596 (clinton +1.0 edge)

#iv. already registered voters 
nc_state['party'].unique() #0 rep, 1 dem 
nc_state['party'].value_counts() #rep=630 (0.436).dem=814 (0.564)
#independent =799 (33.6%)
#clinton score +5.352, trump score +5.032 (clinton +1.53 edge)

#predict if voter favors replican candidate 
#### logistic regression
nc_party=nc_state[(nc_state['party']==0) | (nc_state['party']==1)|(nc_state['party']==2)]
len(nc_party)/len(nc_state) #94.4% either rep, dem, or ind in the sample  
import numpy as np 
import pandas as pd 

#subset data 
nc_party1=nc_party[["sen_fav","gov_fav","vote_lh","gender_x","education","race_card","hispanic",
"age","file_vt16","file_vt14","file_vt12","age","rep_fav","dem_fav"]]
nc_party1.shape #2243,12 

nc_party1['gov_fav']=nc_party1[(nc_party1['gov_fav']==0)|(nc_party1['gov_fav']==1)|(nc_party1['gov_fav']==2)]
nc_party1=nc_party1.dropna(axis=1,how='any') 
nc_party1.shape #2179
nc_party_x=nc_party1[(nc_party1['rep_fav']==1)|(nc_party1['rep_fav']==0)]
nc_party_x.shape 

#split the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)
predict=lr_model.predict(X_test)
accuracy_score(y_test,predict) #83.3% accuracy (whether voter favors trump or not?)

#*******************************
#predicting whether voter has favorable view of clinton or not?
nc_party=nc_state[(nc_state['party']==0) | (nc_state['party']==1)|(nc_state['party']==2)]
len(nc_party)/len(nc_state) #94.4% either rep, dem, or ind in the sample  
nc_party_x1=nc_party1[(nc_party1['dem_fav']==1)|(nc_party1['dem_fav']==0)]
nc_party_x1.shape 
nc_party_x1
X=nc_party_x1.iloc[:,0:11]
y=nc_party_x1.iloc[:,12]

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

lr_model1=LogisticRegression()
lr_model1.fit(X_train,y_train)
predict1=lr_model1.predict(X_test)
accuracy_score(y_test,predict1) #87.4% accuracy(whether voter has favorable view of clinton or not)
X=nc_party_x1.iloc[:,0:11] #features 
y=nc_party_x1.iloc[:,12] #target 

from sklearn.preprocessing import StandardScaler
#standarize the features
X=StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA 
pca=PCA(n_components=11)
principalComponents=pca.fit_transform(X)
principalComponents

from sklearn.preprocessing import scale
X=scale(X)
pca=PCA(n_components=11)
pca.fit(X)
#amount of variance explained
var=pca.explained_variance_ratio_ 
var 

#test and train data
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

lr_model_scale=LogisticRegression()
lr_model_scale.fit(X_train,y_train)
predict2=lr_model_scale.predict(X_test)
accuracy_score(y_test,predict2) #87.4%


#visualizations 
import seaborn 
import pylab 
#gender turnout score
sns.boxplot(x="gender",y="turnout_score",hue="gender",data=nc_state)
pylab.show() 

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#intialize the rng 
seed=893
np.random.seed(seed)

#encode class values as integers
X
y
encoder=LabelEncoder() 
encoder.fit(y)
encoded_y=encoder.transform(y)

#baseline model
def baseline_model():
    #create the model
    model=Sequential()
    model.add(Dense(11,input_dim=11,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    #compile the model?
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model 

#evaluate the keras binary model
est=KerasClassifier(build_fn=baseline_model,epochs=25,batch_size=5,verbose=0)
kfold=StratifiedKFold(n_splits=9,shuffle=True,random_state=seed)
results=cross_val_score(est,X,encoded_y,cv=kfold)
results.mean() #86.36% accuracy 

#1. standardize data -re-scaled so that mean 0, s.d. 1

#define a standard scaler followed by the neural network 
#eval. baseline model con 
np.random.seed(seed)
estimators=[] 
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=baseline_model,epochs=25,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(pipeline,X,encoded_y,cv=kfold)
results.mean() #86.13% accuracy

#create smaller network 
def smaller_baseline():
    model=Sequential()
    model.add(Dense(6,input_dim=11,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasClassifier(build_fn=smaller_baseline,epochs=50,batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)
results.mean() #82.22% 


#create a larger network 
def larger_baseline():
    model=Sequential()
    model.add(Dense(11,input_dim=11,kernel_initializer='normal',activation='relu'))
    model.add(Dense(4,kernel_initializer='normal',activation='sigmoid'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=larger_baseline, epochs=1, batch_size=1, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)) #84.7% accuracy 












