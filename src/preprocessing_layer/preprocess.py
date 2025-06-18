import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, numerical_col, ordinal_col, categorical_col):
    """
    Preprocess the dataset: clip numerical values, handle missing values, and scale numerical features.
    
    Args:
        data (pd.DataFrame): Input dataset.
        numerical_col (list): List of numerical column names.
        ordinal_col (list): List of ordinal column names.
        categorical_col (list): List of categorical column names.
        
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Clip numerical values to reasonable ranges
    data['Age'] = data['Age'].clip(lower=0)
    data['Blood_Pressure'] = data['Blood_Pressure'].clip(50, 200)
    data['BMI'] = data['BMI'].clip(15, 50)
    data['Heart_Rate'] = data['Heart_Rate'].clip(30, 200)
    data['Sleep_Hours'] = data['Sleep_Hours'].clip(0, 12)
    data['Exercise_Hours'] = data['Exercise_Hours'].clip(0, 10)
    data['Water_Intake'] = data['Water_Intake'].clip(0, 10)
    data['Stress_Level'] = data['Stress_Level'].clip(0, 10)
    
    # Handle missing values
    for col in numerical_col:
        data[col] = data[col].fillna(data[col].median())
    for col in ordinal_col:
        data[col] = data[col].fillna(data[col].mode()[0])
    for col in categorical_col:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_col] = scaler.fit_transform(data[numerical_col])
    
    return data
