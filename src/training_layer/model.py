from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Logistic Regression model and return train/test split data.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Proportion of data for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def predict(model, X_test):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained LogisticRegression model.
        X_test (pd.DataFrame): Test features.
        
    Returns:
        np.ndarray: Predicted labels.
    """
    return model.predict(X_test)