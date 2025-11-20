import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/ADMIN/Desktop/Customer Churn/null_value_treated.csv')
print(df.columns)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
y=df['Churn']
x=df[['MonthlyCharges','TotalCharges']]
x.fillna(x.mean(),inplace=True)
print(x.isna().sum())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2
,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred)
#### model evaluation
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred) 
print("Accuracy:",accuracy)
print(conf_matrix)

##### classification report
class_report=classification_report(y_test,y_pred)
print(class_report)


#### hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors':np.arange(1,25),'weights':['uniform','distance'],
            'metric':['euclidean','manhattan','minkowski']}
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid,cv=5)
grid.fit(x_train,y_train)
print("Best parameters:",grid.best_params_)
best_knn=grid.best_estimator_
y_pred_best=best_knn.predict(x_test)
from sklearn.metrics import accuracy_score
best_accuracy=accuracy_score(y_test,y_pred_best)
print("Best accuracy after hyperparameter tuning:",best_accuracy)



#### save the model
import joblib
joblib.dump(best_knn,'knn_model.pkl')

### load the model
loaded_model=joblib.load('knn_model.pkl')
y_loaded_pred=loaded_model.predict(x_test)
print("Predictions from loaded model:",y_loaded_pred)
pd.DataFrame(y_loaded_pred,columns=['Churn_Prediction']).to_csv('churn_predictions.csv',index=False)


#### build a streamlit app for this model and run the app
import streamlit as st
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict churn:")
monthly_charges=st.number_input("Monthly Charges",min_value=0.0,max_value=200.0,value=70.0)
total_charges=st.number_input("Total Charges",min_value=0.0,max_value
=10000.0,value=2000.0)
if st.button("Predict Churn"):  
    input_data=np.array([[monthly_charges,total_charges]])
    prediction=loaded_model.predict(input_data)
    if prediction[0]==1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
    