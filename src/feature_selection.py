import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def feature_selection(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LASSO Regression with cross-validation
    lasso = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Get the coefficients and selected features
    coef = lasso.coef_
    selected_features = np.where(coef != 0)[0]
    
    print(f"Selected features indices: {selected_features}")
    print(f"Feature coefficients: {coef[selected_features]}")
    
    return selected_features

if __name__ == "__main__":
    # Load preprocessed data
    data = pd.read_csv("data/ckc.csv")  # Adjust the path as necessary
    
    # Handle missing or invalid values in the target column
    X = data.iloc[:, :-1]  
    y = data['Diagnosis']
    
    y = y.replace({'Confidential': 1, 'Not Confidential': 0})
    y = pd.to_numeric(y, errors='coerce') 
    y = y.dropna()
    X = X.loc[y.index]
    
    # Perform feature selection using LASSO
    selected_features = feature_selection(X, y)
    # Get feature names from the original dataset
    feature_names = X.columns
    # Get the selected feature names
    selected_feature_names = feature_names[selected_features]

    # Convert to DataFrame
    selected_features_df = X[selected_feature_names]

    # Save to CSV
    selected_features_df.to_csv("data/selected_features.csv", index=False)
    
    df=pd.read_csv("data/selected_features.csv")
    df['Diagnosis']=data['Diagnosis']
    df.to_csv("data/selected_features.csv", index=False) 

    print("Feature selection completed.")
