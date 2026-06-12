import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_data():
    df=pd.read_csv(RAW_DATA_PATH)
    return df

def drop_unnecessary_cols(df):
    df=df.drop(columns=["Unnamed: 0","flight"])
    return df

def outliers_handling(df):
    col="duration"
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    IQR=q3-q1
    lower=q1-(1.5*IQR)
    upper=q3+(1.5*IQR)
    df[col]=df[col].clip(lower,upper)
    return df

def encoding(df):
    le=LabelEncoder()
    df["stops"]=df["stops"].map({"zero":0,"one":1,"two_or_more":2})
    df["class"]=df["class"].map({"Economy":0,"Business":1})
    cat_cols=["airline","source_city","departure_time","arrival_time","destination_city"]
    for col in cat_cols:
        df[col]=le.fit_transform(df[col])
    return df

def scaler_features(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train) 
    X_test_scaled=scaler.transform(X_test)
    
    os.makedirs(MODELS_PATH,exist_ok=True)
    joblib.dump(scaler,os.path.join(MODELS_PATH,"scaler.pkl"))
    
    return X_train_scaled,X_test_scaled

def save_processed_data(X_train,X_test,Y_train,Y_test):
    os.makedirs(PROCESSED_PATH,exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(PROCESSED_PATH,"X_train.csv"),index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(PROCESSED_PATH,"X_test.csv"),index=False)
    pd.DataFrame(Y_train).to_csv(os.path.join(PROCESSED_PATH,"Y_train.csv"),index=False)
    pd.DataFrame(Y_test).to_csv(os.path.join(PROCESSED_PATH,"Y_test.csv"),index=False)
    
def preprocess():
    df=load_data()
    df=drop_unnecessary_cols(df)
    df=outliers_handling(df)
    df=encoding(df)
    
    X=df.drop(columns=[TARGET_COL])
    Y=df[TARGET_COL]
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    
    X_train,X_test=scaler_features(X_train,X_test)
    save_processed_data(X_train,X_test,Y_train,Y_test)
    print("Preprocessing completed....")
    return X_train,X_test,Y_train,Y_test



if __name__=="__main__":
    preprocess()