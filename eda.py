import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
df=pd.read_csv('C:/Users/ADMIN/Desktop/Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
### only graphs and save each graph in folder location called images
def  visualize_data(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Count')
    plt.savefig('images/churn_count.png')
    plt.figure(figsize=(10,6))
    sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
    plt.title('Monthly Charges Distribution')
    plt.savefig('images/monthly_charges_distribution.png')
    plt.figure(figsize=(10,6))
    sns.histplot(df['TotalCharges'].dropna(), bins=30, kde=True)
    plt.title('Total Charges Distribution')
    plt.savefig('images/total_charges_distribution.png')

visualize_data(df)