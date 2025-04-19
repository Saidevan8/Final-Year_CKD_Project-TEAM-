import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data/ckc.csv')
X = data.drop(columns=['Diagnosis'])  # Replace 'Diagnosis' with the target column
y = data['Diagnosis']  # Replace 'Diagnosis' with the target column

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Address class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),
    'LightGBM': LGBMClassifier(),
    'LSVM': SVC(kernel='linear', probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Define a function to build a simple DNN model
def build_dnn(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else float('nan')

    # Append results
    results.append([name, accuracy, precision, recall, f1, roc_auc])

# Train and evaluate DNN
dnn_model = build_dnn(X_train.shape[1])
dnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
y_pred_dnn = (dnn_model.predict(X_test) > 0.5).astype("int32")

# Calculate DNN metrics
accuracy = accuracy_score(y_test, y_pred_dnn)
precision = precision_score(y_test, y_pred_dnn)
recall = recall_score(y_test, y_pred_dnn)
f1 = f1_score(y_test, y_pred_dnn)
roc_auc = roc_auc_score(y_test, dnn_model.predict(X_test))

# Append DNN results
results.append(['DNN', accuracy, precision, recall, f1, roc_auc])

# Create a DataFrame for comparison
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

# Display the results
print(results_df)

# Save results to CSV
results_df.to_csv('data/model_comparison.csv', index=False)
