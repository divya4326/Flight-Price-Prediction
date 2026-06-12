import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_processed_data():
    X_train =pd.read_csv(os.path.join(PROCESSED_PATH,"X_train.csv"))
    X_test =pd.read_csv(os.path.join(PROCESSED_PATH,"X_test.csv"))
    Y_train =pd.read_csv(os.path.join(PROCESSED_PATH,"Y_train.csv")).squeeze()
    Y_test =pd.read_csv(os.path.join(PROCESSED_PATH,"Y_test.csv")).squeeze()
    return X_train,X_test,Y_train,Y_test

def get_models():
    return {
        'linearRegression' : LinearRegression(),
        'ridge'            : Ridge(alpha=RIDGE_ALPHA),
        'lasso'            : Lasso(alpha=LASSO_ALPHA),
        'Decisiontree'     : DecisionTreeRegressor(random_state=RANDOM_STATE),
        'RandomForest'     : RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,random_state=RANDOM_STATE),
        'lightbgm'         : LGBMRegressor(n_estimators=LGBM_N_ESTIMATORS,learning_rate=LGBM_LEARNING_RATE,random_state=RANDOM_STATE)
    
    }
    
def train_models(models,X_train,Y_train):
    trained_models={}
    for name,model in models.items():
        print(f"Training {name}..")
        model.fit(X_train,Y_train)
        trained_models[name]=model
    return trained_models

def save_models(trained_models):
    os.makedirs(MODELS_PATH,exist_ok=True)
    for name,model in trained_models.items():
        path=os.path.join(MODELS_PATH,f"{name}.pkl")
        joblib.dump(model,path)        
def train():
    X_train,X_test,Y_train,Y_test=load_processed_data()
    models=get_models()
    trained_models=train_models(models,X_train,Y_train)
    save_models(trained_models)
    print("Training Completed")
    return trained_models 

if __name__=="__main__":
    train()