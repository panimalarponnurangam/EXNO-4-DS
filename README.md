# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
    
   ````
import pandas as pd
from scipy import stats
import numpy as np
````
````
df= pd.read_csv("/content/bmi.csv")
df
````
![image](https://github.com/user-attachments/assets/6ab1248e-200e-426d-82ed-761bc991384f)
````
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
````
![image](https://github.com/user-attachments/assets/bd9744c6-5541-4d0a-8171-878a159d426a)

````
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
````
![image](https://github.com/user-attachments/assets/63dcd83f-534c-4b11-a0ee-70eb062d6a68)
````
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

````
![image](https://github.com/user-attachments/assets/acb4a137-7985-4552-b0f3-8a2081d99fa9)
````
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
````
![image](https://github.com/user-attachments/assets/4f38e0ed-b602-4e12-b390-9a661d8860eb)
````
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
````
![image](https://github.com/user-attachments/assets/e73b51cd-e079-487e-bf71-9119a773eba1)
````
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
````
![image](https://github.com/user-attachments/assets/88dd43a0-b0ac-41f9-b236-c47621b006bc)

````
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
````
````
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
````
![image](https://github.com/user-attachments/assets/0aa9102e-0a67-4b08-a8b7-71d13f23fe95)
````
data.isnull().sum()
````
![image](https://github.com/user-attachments/assets/ca3a15a5-0c88-4e82-b2bf-7cc79e91194a)
````
missing=data[data.isnull().any(axis=1)]
missing
````
![image](https://github.com/user-attachments/assets/75a30230-222c-41f7-bfea-2360289cdc4a)

````
data2=data.dropna(axis=0)
data2
````

![379088554-b4c4a3bc-29ea-414a-87f3-b55bdc247b91](https://github.com/user-attachments/assets/5dca8781-6cc0-4ad1-8046-20bf91c5054f)
```
sal=data['SalStat']
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![379088782-dd6ae99d-38d6-43a9-b2e1-5a65cd23b2c7](https://github.com/user-attachments/assets/2521577f-2e3a-4e4b-9bbf-bc6a195b42d1)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![379088937-9b2cfebc-fcfa-4e2d-905c-15008f45a0a0](https://github.com/user-attachments/assets/4b4a13b1-9881-46ae-9e67-e6f3d98ae0a3)

```
data2
```
![379089115-44494a39-c872-4c7a-8107-45ce2c819c25](https://github.com/user-attachments/assets/3187c950-c629-427a-bd57-d8e054cfa21e)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![379089273-5c839ffb-9026-43be-acd3-80cc0bf5d2af](https://github.com/user-attachments/assets/4ba89541-9992-40ec-8f04-363f4cb5714f)

```
columns_list=list(new_data.columns)
print (columns_list)
```
![379089526-71ac1b70-7f9d-4f07-a83e-81fc312c4a22](https://github.com/user-attachments/assets/0d4c465e-bbe5-464e-8026-2d6a2361c2ee)

```
features=list(set(columns_list))
print(features)
```
![379089761-5efa78d8-f82f-40a9-9744-011e342d48cd](https://github.com/user-attachments/assets/17f9f87a-c1b4-4ab8-9c24-218c53315cb3)
```
y=new_data['SalStat'].values
print(y
```
![379089878-c6b1eccf-c27e-472e-9e93-cd1f8978a829](https://github.com/user-attachments/assets/a0e8f85e-1af7-4cd7-a0b8-49bc05d5a5fe)

```
x=new_data[features].values
print(x)
```

![379090052-4c3069a9-0fb1-4cb4-a2fd-731f01a764a9](https://github.com/user-attachments/assets/014cd356-33ec-40cb-8e77-5a26bd1bdf8d)
```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```

![379090377-cf746ca8-2502-437b-a4ee-c7eb600a41f3](https://github.com/user-attachments/assets/48574d19-4506-4d05-bd92-e98b641f1f98)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![379090601-2ce335c8-01f3-4123-9ab7-4a854646dfe9](https://github.com/user-attachments/assets/c8390991-1260-4e6c-8233-e3d383272342)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![379090727-8ed42482-9265-4700-8622-72e21f3aa981](https://github.com/user-attachments/assets/af516bee-ea27-42ef-8a35-28a38085d859)
```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
````
![379090844-9b71dfdf-f169-473b-b411-a6fd98d583db](https://github.com/user-attachments/assets/80b26868-b188-4253-83bb-756bd53f9687)

```
data.shape
```
![379090981-55fa3451-ad3d-425a-bc45-e3ce186e20b6](https://github.com/user-attachments/assets/8cd7446f-3b2b-4f52-951e-b62eff236f07)





# RESULT:
          Thus perform Feature Scaling and Feature Selection process and save the 
          data to a file successfully.
