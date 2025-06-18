import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on model coefficients.
    
    Args:
        model: Trained LogisticRegression model.
        feature_names (list): List of feature names.
    """
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Logistic Regression Coefficient (Absolute Value)')
    plt.title('Feature Importance for Classification')
    plt.tight_layout()
    plt.show()