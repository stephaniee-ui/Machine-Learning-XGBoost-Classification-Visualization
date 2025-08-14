import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

file_path = "Cleaned_CA_Env_Data.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month

label_encoders = {}
for col in ['Stn_Name', 'CIMIS_Region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


drop_cols = ['Date']
df.drop(columns=drop_cols, inplace=True)
#df.fillna(df.median(), inplace=True)

print(df)

X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handling imbalance using SMOTE
#smote = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

xgb_model = XGBClassifier(
    scale_pos_weight=20,
    max_depth=6,
    learning_rate=0.3,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=1, 
    eval_metric='logloss',
    random_state=1
)
#xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_model.fit(X_train, y_train)

y_prob = xgb_model.predict_proba(X_test)[:, 1]
print(y_prob)
y_pred = (y_prob > 0.7).astype(int)

precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob) 
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")