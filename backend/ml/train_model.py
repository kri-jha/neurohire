import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import glob

def load_dataset(dataset_path):
    """
    Load dataset from the specified path.
    Handles CSV files directly.
    """
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        # For CSV files
        if os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
            print(f"Loading CSV file: {dataset_path}")
            df = pd.read_csv(dataset_path)
            print(f"CSV loaded successfully with {len(df)} rows")
            return df
            
        # For directory containing CSV files
        elif os.path.isdir(dataset_path):
            # First check for CSV files
            csv_files = glob.glob(os.path.join(dataset_path, "**/*.csv"), recursive=True)
            if csv_files:
                # Load the largest CSV file
                csv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                csv_path = csv_files[0]
                print(f"Loading CSV file: {os.path.basename(csv_path)}")
                df = pd.read_csv(csv_path)
                print(f"CSV loaded successfully with {len(df)} rows")
                return df
            else:
                raise ValueError(f"No CSV files found in {dataset_path}")
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def preprocess_data(data):
    """
    Preprocess the data for training.
    """
    print("Preprocessing data...")
    
    # Limit dataset size to avoid memory issues
    if len(data) > 5000:
        print(f"Limiting dataset to 5000 rows (from {len(data)})")
        data = data.sample(5000, random_state=42)
    
    # Identify text columns and numeric columns
    text_cols = data.select_dtypes(include=['object']).columns.tolist()
    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Found {len(text_cols)} text columns and {len(num_cols)} numeric columns")
    
    # Use the first text column as target if available, otherwise use first column
    if text_cols:
        target_col = text_cols[0]
        # Limit feature columns to avoid memory issues
        feature_cols = (text_cols[1:3] + num_cols[:5]) if len(text_cols) > 1 else num_cols[:5]
    else:
        target_col = data.columns[0]
        feature_cols = data.columns[1:6]  # Limit to 5 features
    
    print(f"Using target column: {target_col}")
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")
    
    # Handle categorical target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[target_col].astype(str))
    
    # Handle features - convert text to dummy variables
    # Limit the number of categories per column to avoid memory issues
    X_parts = []
    for col in feature_cols:
        if data[col].dtype == 'object':
            # For text columns, limit to top 10 categories
            top_cats = data[col].value_counts().nlargest(10).index
            data[col] = data[col].apply(lambda x: x if x in top_cats else 'Other')
        X_parts.append(pd.get_dummies(data[col], prefix=col, drop_first=True))
    
    X = pd.concat(X_parts, axis=1)
    
    # Fill missing values
    X = X.fillna(0)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def train_model(X_train, y_train):
    """
    Train a machine learning model.
    """
    print("Training model...")
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, scaler, label_encoder, output_dir):
    """
    Save the trained model and preprocessing objects.
    """
    print(f"Saving model to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save the label encoder
    encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(label_encoder, encoder_path)
    
    print("Model saved successfully!")

def main(args):
    """
    Main function to run the training pipeline.
    """
    # Load the dataset
    data = load_dataset(args.dataset_path)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model
    save_model(model, scaler, label_encoder, args.output_dir)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model.')
    
    args = parser.parse_args()
    
    # Run the training pipeline
    main(args)