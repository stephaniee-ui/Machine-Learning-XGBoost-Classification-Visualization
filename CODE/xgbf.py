import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, roc_auc_score, f1_score, make_scorer
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

df.drop(columns=['Date'], inplace=True)

X = df.drop(columns=['Target'])
y = df['Target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

xgb_model = XGBClassifier(
    scale_pos_weight=20,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=1
)

param_grid = {
    'learning_rate': [0.05, 0.1, 0.3, 0.5, 0.75],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

scoring = make_scorer(f1_score)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=scoring,
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:\n", grid_search.best_params_)

y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.7).astype(int)

precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
