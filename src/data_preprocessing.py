import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display initial information about the data
    print("Initial Data Overview:")
    print(data.head())
    print(data.info())
    print(data.isnull().sum())
    
    # Separate features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Feature scaling and encoding
    scaler = StandardScaler()
    encoder = OneHotEncoder()

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', num_imputer), ('scaler', scaler)]), numerical_columns),
            ('cat', Pipeline(steps=[('imputer', cat_imputer), ('encoder', encoder)]), categorical_columns)
        ]
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    print(data.info())
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    file_path = "data/ckc.csv"  # Adjust path for your dataset
    target_column = "Diagnosis"  # Replace with the actual target column name
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, target_column)
    print("Data preprocessing completed.")
