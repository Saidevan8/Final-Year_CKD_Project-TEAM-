import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
model_paths = {
    'XGBoost': r'models\m1\xgb_model.joblib',
    'CatBoost': r'models\m2\catboost_model.joblib',
    'LightGBM': r'models\m3\lgb_model.joblib',
    'RandomForest': r'models\m4\rf_model.joblib',
    'LSVM': r'models\m5\lsvm_model.joblib',
    'DNN': r'models\m6\dnn_model.keras',
    'LogisticRegression': r'models\m7\lr_model.joblib',
    'KNN': r'models\m8\knn_model.joblib',
    'GradientBoosting': r'models\m9\gb_model.joblib',
    'NaiveBayes': r'models\m10\nb_model.joblib',
    'Scaler': r'models\scaler.joblib'
}


# Ensure model directories exist
os.makedirs(os.path.dirname(model_paths['DNN']), exist_ok=True)

# Load dataset
data = pd.read_csv(r'data\ckc.csv')
X = data.iloc[:, :-1] 
Y = data['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
lasso.fit(X_scaled, Y)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, Y)

# Address class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter grids
param_grids = {
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    'CatBoost': {'depth': [4, 6], 'learning_rate': [0.1, 0.01]},
    'LightGBM': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'LSVM': {'C': [0.1, 1], 'kernel': ['linear']},
    'LogisticRegression': {'C': [0.1, 1]},
    'KNN': {'n_neighbors': [3, 5]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
}

# Initialize models
models = {
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(silent=True),
    'LightGBM': LGBMClassifier(),
    'RandomForest': RandomForestClassifier(),
    'LSVM': SVC(probability=True),
    'LogisticRegression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'NaiveBayes': GaussianNB()
}

# Hyperparameter tuning and model saving
for name, model in models.items():
    print(f"Tuning {name}...")
    if name in param_grids:
        grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring='accuracy', cv=3)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    
    # Save the tuned model
    dump(best_model, model_paths[name])

# Evaluate all models
results = []
for name, model in models.items():
    if name == 'Scaler':
        continue
    print(f"Evaluating {name}...")
    m=model
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, accuracy,precision,recall,f1])

# Save evaluation results
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy','Precision','Recall','F1-score'])
results_df.to_csv(r'data/after_hyper_parameter_tuning.csv', index=False)
print("Hyperparameter tuning and evaluation complete!")
