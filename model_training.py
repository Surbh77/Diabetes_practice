from cgi import test
import pandas as pd
import numpy as np
import pickle 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split    

df=pd.DataFrame(pd.read_csv('diabetes.csv'))
# print(df.isna().sum())

x=df.drop('Outcome',axis=1)
y=df['Outcome']


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=24,test_size=0.2,stratify=y)


dct=DecisionTreeClassifier()
dct_model=dct.fit(x_train,y_train)

y_pred_train=dct_model.predict(x_train)
acc_scc_train=accuracy_score(y_pred_train,y_train)
print(acc_scc_train)


y_pred_test=dct_model.predict(x_test)
acc_scc_test=accuracy_score(y_pred_test,y_test)
print(acc_scc_test)

pickle.dump(dct_model,open('diabetes.pkl','wb'))