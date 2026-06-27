import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("--- Starting Astronomical Classification Test ---")
    
    # 1. Loading the Test Data
    # The script runs from the 'src' directory, so we navigate out and into 'date'
    data_path = '../date/test_data.csv' 
    print(f"Loading test data from '{data_path}'...")
    try:
        test_df = pd.read_csv(data_path) 
    except FileNotFoundError:
        print(f"Error: Could not find '{data_path}'. Please verify the folder structure.")
        return

    # Separate the features from the target variable (labels)
    X_test = test_df.drop('class_encoded', axis=1)
    y_test = test_df['class_encoded']

    # 2. Loading the pre-trained winning model
    # The model is saved in the exact same directory (src) as this script
    model_path = 'astronomy_rf_model.pkl'
    print(f"Loading the trained model from '{model_path}'...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Did you run the save block in Jupyter?")
        return

    # 3. Running Predictions
    print("Running predictions on unseen test data...")
    y_pred = model.predict(X_test)

    # 4. Printing Final Results
    acc = accuracy_score(y_test, y_pred)
    print("\n=========================================")
    print(f"FINAL TEST ACCURACY: {acc * 100:.2f}%")
    print("=========================================\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['STAR (0)', 'GALAXY (1)', 'QSO (2)']))

if __name__ == "__main__":
    main()