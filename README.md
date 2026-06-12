# Flight-Price-Prediction

A machine learning project that predicts Indian domestic flight prices based on airline, route, timing, and booking details. Built with a modular pipeline covering data preprocessing, multi-model training, evaluation, and explainability.

---

## Dataset

- **Source**: Indian flight prices dataset (Kaggle)
- **Size**: ~300,000 rows
- **Target**: Flight price (₹)

---

##  Models Trained

| Model              | R2     |   RMSE   | MAE     |
----------------------------------------------------
| Linear Regression  | 0.9063 | 6951.11  | 4527.77 |
| Ridge Regression   | 0.9063 | 6951.11  | 4527.78 |
| Lasso Regression   | 0.9063 | 6951.10  | 4527.36 |
| Decision Tree      | 0.9754 | 3563.17  | 1178.76 |
| Random Forest      | 0.9849 | 2786.35  | 1091.31 |
| LightGBM           | 0.9692 | 3987.78  | 2360.47 |

 **Best Model: Random Forest** with R2 = 0.9849

---

## Project Structure
Flight-Price-Prediction/

├── data/

│   ├── raw/                  # Original dataset (not tracked)

│   └── processed/            # Train/test splits (not tracked)

├── models/                   # Saved model .pkl files (not tracked)

├── notebooks/

│   └── eda.ipynb             # Exploratory Data Analysis

├── reports/

│   └── model_comparison.csv  # Evaluation results

├── src/

│   ├── init.py

│   ├── preprocessing.py      # Data cleaning, encoding, scaling

│   ├── train.py              # Model training

│   ├── evaluate.py           # Model evaluation

│   ├── predict.py            # Single prediction

│   └── shap_analysis.py      # SHAP explainability

├── config.py                 # Central configuration

├── main.py                   # Full pipeline runner

└── requirements.txt

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/divya4326/Flight-Price-Prediction.git
cd Flight-Price-Prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Place `Clean_Dataset.csv` in `data/raw/`

### 5. Run full pipeline
```bash
python main.py
```

---

## 🔍 Features Used

| Feature | Type | Encoding |
|---|---|---|
| airline | Categorical | Label Encoding |
| source_city | Categorical | Label Encoding |
| destination_city | Categorical | Label Encoding |
| departure_time | Categorical | Label Encoding |
| arrival_time | Categorical | Label Encoding |
| stops | Ordinal | Manual Mapping (0/1/2) |
| class | Ordinal | Manual Mapping (0/1) |
| duration | Numerical | StandardScaler |
| days_left | Numerical | StandardScaler |

---

##  Key Decisions

- **Dropped `flight` column** — unique flight codes have no predictive signal
- **Outliers in `duration` capped** using IQR — 2110 extreme values treated
- **Outliers in `price` kept** — real expensive flights, model must learn them
- **Scaler fitted on train only** — prevents data leakage
- **Random Forest selected** as best model for prediction and SHAP analysis

---

## Tech Stack

- Python 3.13
- pandas, numpy
- scikit-learn
- Machine Learning
- matplotlib, seaborn

---

##  Author

**Divya Chaganti**  
B.Tech AI, ML  — Geethanjali College of Engineering and Technology  
[GitHub](https://github.com/divya4326)