## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 

df=pd.read_csv("Encoding Data.csv")
print(df)
```

<img width="367" height="217" alt="image" src="https://github.com/user-attachments/assets/5bc2a492-2dc1-4eca-a740-de9208a6c92e" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="370" height="202" alt="image" src="https://github.com/user-attachments/assets/7ea9ac80-8b3b-41a8-a475-37c9ad16e56e" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
print(df)
```
<img width="403" height="231" alt="image" src="https://github.com/user-attachments/assets/bcfe99c9-c41f-4216-b16b-3f9f46bb5cee" />

```
le=LabelEncoder()

dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
print(dfc)
```

<img width="366" height="222" alt="image" src="https://github.com/user-attachments/assets/0e2bde90-7636-42fb-a030-05c308c5c0e1" />

```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
df2=pd.concat([df2,enc],axis=1)
print(df2)
```

<img width="492" height="223" alt="image" src="https://github.com/user-attachments/assets/148b50b2-cfca-4be3-87e1-db3af1ea1cbb" />

```
pd.get_dummies(df2,columns=["nom_0"])

```
<img width="650" height="318" alt="image" src="https://github.com/user-attachments/assets/d35c01e7-9243-4dcf-8be6-a9383fc2a05e" />

```
pip install --upgrade category_encoders

```
<img width="1120" height="471" alt="image" src="https://github.com/user-attachments/assets/181eba13-4c06-4911-b38f-9741b9623786" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="505" height="306" alt="image" src="https://github.com/user-attachments/assets/f2f22b4c-4dad-4a3c-b733-e5934b99b60b" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="707" height="313" alt="image" src="https://github.com/user-attachments/assets/df6e457e-b734-4c23-9cdf-a18f9cf406ab" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder()
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1) 
CC
```

<img width="635" height="311" alt="image" src="https://github.com/user-attachments/assets/83e72219-66a8-48f0-bafe-7b6f1bfd8ccb" />

```
import pandas as pd 
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="725" height="372" alt="image" src="https://github.com/user-attachments/assets/d7d073f8-89d6-4ea1-836e-510d1f84eba7" />

```
df.skew()
```
<img width="392" height="98" alt="image" src="https://github.com/user-attachments/assets/d091e36a-86be-4ef3-af31-59bf6f285213" />

```
np.log(df["Highly Positive Skew"])
```
<img width="622" height="215" alt="image" src="https://github.com/user-attachments/assets/78eda63e-76b8-4eae-b64a-cfadcba3dda5" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="562" height="220" alt="image" src="https://github.com/user-attachments/assets/d610f572-9ffc-4804-9e13-366549d61fea" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="556" height="212" alt="image" src="https://github.com/user-attachments/assets/8ef90448-eb1e-48ae-8f4b-bacb03225773" />

```
np.square(df["Highly Positive Skew"])
```
<img width="560" height="227" alt="image" src="https://github.com/user-attachments/assets/f7eee031-a7d0-4a71-9803-77f64f1fbe5c" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1080" height="372" alt="image" src="https://github.com/user-attachments/assets/86bffd09-5831-4041-80b7-3ce2b48a7cc3" />

```
df.skew()
```
<img width="470" height="133" alt="image" src="https://github.com/user-attachments/assets/497a3fba-0575-4205-8c05-1b86c1491ce1" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="520" height="142" alt="image" src="https://github.com/user-attachments/assets/629893fb-e749-47e5-a138-a1aa6c82dfb2" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1107" height="398" alt="image" src="https://github.com/user-attachments/assets/90b5a8f8-6e73-4519-96ae-18ee350ef4fd" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="579" height="432" alt="download" src="https://github.com/user-attachments/assets/f3fff8ec-973e-4e29-b385-ec61b8029d7e" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')  
plt.show()
```
<img width="574" height="432" alt="download" src="https://github.com/user-attachments/assets/de4673c4-b813-4c87-972f-188e60893a7f" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/58764964-9d88-4b51-aa53-bf63035654ad" />

# RESULT:
  Thus the Feature Encoding and Transformation process has been done for the given data.

       
