import os
#===============PATHS================================
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH=os.path.join(BASE_DIR,"data/raw/Clean_Dataset.csv")
PROCESSED_PATH=os.path.join(BASE_DIR,"data/processed")
MODELS_PATH=os.path.join(BASE_DIR,"models/")
REPORTS_PATH=os.path.join(BASE_DIR,"reports/")

#====================Data=======================
TARGET_COL="price"
TEST_SIZE=0.2
RANDOM_STATE=42

#===================Features====================
NUMERICAL_COLS=['duration', 'days_left']
CATEGORICAL_COLS=['airline', 'flight', 'source_city', 'departure_time', 'stops',
       'arrival_time', 'destination_city', 'class']

#==================Models==========================
MODELS_TO_TRAIN = [
    "linearRegression",
    "ridge",
    "lasso",
    "Decisiontree",
    "RandomForest",
    "lightbgm"
]

#================Hyperparameters=============
RIDGE_ALPHA        = 1.0
LASSO_ALPHA        = 1.0
RF_N_ESTIMATORS    = 100
LGBM_N_ESTIMATORS  = 100
LGBM_LEARNING_RATE = 0.1



