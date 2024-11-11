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


# RESULT:
       # INCLUDE YOUR RESULT HERE
