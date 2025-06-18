from data_layer.data_loader import load_data
from preprocessing_layer.preprocess import preprocess_data
from training_layer.model import train_model, predict
from evaluation_layer.evaluate import evaluate_model
from visualization_layer.visualize import plot_feature_importance

def main():
    # Define column types
    numerical_col = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level', 'Heart_Rate', 
                     'Sleep_Hours', 'Exercise_Hours', 'Water_Intake', 'Stress_Level']
    ordinal_col = ['Smoking', 'Alcohol', 'Diet', 'MentalHealth', 'PhysicalActivity', 
                   'MedicalHistory', 'Allergies']
    categorical_col = ['Diet_Type_Vegan', 'Diet_Type_Vegetarian', 'Blood_Group_AB', 
                      'Blood_Group_B', 'Blood_Group_O']
    target_col = 'Target'
    
    # Load data
    data = load_data()
    
    # Preprocess data
    data = preprocess_data(data, numerical_col, ordinal_col, categorical_col)
    
    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Train model and get split data
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Make predictions
    y_pred = predict(model, X_test)
    
    # Evaluate model
    evaluate_model(y_test, y_pred)
    
    # Plot feature importance
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    main()
    