
import pandas as pd

columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
rawdata=pd.read_csv(r"dataset\heartdisease\processed.hungarian.data", names=columns,encoding='latin1', na_values='?')


# print(rawdata.duplicated().sum())

# print(rawdata.head())
# print(rawdata.shape)
rawdata.drop_duplicates(inplace=True)

# print(rawdata.shape)
# print(rawdata.isnull().sum())

# print(rawdata.columns)

# print((rawdata['age'].isnull().sum())/rawdata.shape[0]*100)

def null_value(i):
    return (rawdata[i].isnull().sum())/rawdata.shape[0]*100


for i in rawdata.columns:
    if null_value(i)>55:
        # print(i, null_value(i))
        rawdata.drop(i,axis=1,inplace=True)


# print(rawdata.mode().iloc[0])
rawdata .fillna(rawdata.mode().iloc[0],inplace=True)

# print(rawdata.shape)

# print(rawdata.isnull().sum())


import matplotlib.pyplot as plt 
import seaborn as sns

# sns.pairplot(rawdata, hue='target')
# plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# print(rawdata.describe())
output=rawdata.pop('target')
input=rawdata

se=StandardScaler()
se.fit(input)
input=pd.DataFrame(se.transform(input), columns=input.columns)

x_train, x_test, y_train, y_test=train_test_split(input, output, test_size=0.3, random_state=42)

lr=LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)) 


import joblib

joblib.dump(lr, 'heart_model.pkl')
