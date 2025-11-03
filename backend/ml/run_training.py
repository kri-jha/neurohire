import subprocess
import os
import sys

def main():
    # Default dataset path from user input
    dataset_path = r"C:\Users\adars\Downloads\archive (1)"
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Starting model training with dataset from: {dataset_path}")
    print("This may take some time depending on the size of your dataset...")
    
    # Run the training script
    try:
        subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_model.py"),
            "--dataset_path", dataset_path,
            "--output_dir", models_dir
        ], check=True)
        
        print("\nTraining completed successfully!")
        print(f"Model saved in: {models_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()