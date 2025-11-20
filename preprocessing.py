import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('C:/Users/ADMIN/Desktop/Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
def preprocess_data(df):
    ## check null value count
    null_counts = df.fillna(np.nan)
    le=LabelEncoder()
    le.fit(df['Churn'])
    df['Churn']=le.transform(df['Churn'])
    return null_counts  

print(preprocess_data(df))
null_values = preprocess_data(df)
null_values.to_csv('null_value_treated.csv', index=True)