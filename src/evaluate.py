import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_processed_data():
    X_test=pd.read_csv(os.path.join(PROCESSED_PATH,"X_test.csv"))
    Y_test=pd.read_csv(os.path.join(PROCESSED_PATH,"Y_test.csv"))
    return X_test,Y_test

def load_models():
    models={}
    for name in MODELS_TO_TRAIN:
        path=os.path.join(MODELS_PATH,f"{name}.pkl")
        models[name]=joblib.load(path)
    return models

def evaluate_models(models,X_test,Y_test):
    result=[]
    for name,model in models.items():
        y_pred=model.predict(X_test)
        r2=r2_score(Y_test,y_pred)
        rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
        mae=mean_absolute_error(Y_test,y_pred)
        result.append({
            "model" : name,
            "R2"    : round(r2,4),
            "RMSE"  : round(rmse,2),
            "MAE"   : round(mae,2) 
        })
        print(f"{name:20s} | R2 :{r2 : .4f} | RMSE :{rmse : .2f} | MAE :{mae: .2f}")
    return pd.DataFrame(result).sort_values("R2",ascending=False)

def save_result(result):
    os.makedirs(REPORTS_PATH,exist_ok=True)
    path=os.path.join(REPORTS_PATH,"model_comparision.csv")
    result.to_csv(path,index=False)
    print("Results Saved..")      
    
def evaluate():
    X_test,Y_test=load_processed_data()
    models=load_models()
    results=evaluate_models(models,X_test,Y_test)
    save_result(results)
    return results
    
if __name__=="__main__":
    evaluate()