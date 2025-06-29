import pandas as pd
import os

def preprocess_data(input_csv, output_csv):
    """
    Loads raw data, fills missing values, and saves a cleaned dataset.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Fill missing values using forward fill method
    df.ffill(inplace=True)


    # Save the cleaned dataset
    df.to_csv(output_csv, index=False)
    print(f"[✓] Processed data saved to: {output_csv}")

if __name__ == "__main__":
    # Define paths
    raw_path = "data/raw/sample_data.csv"
    cleaned_path = "data/processed/cleaned_data.csv"

    # Run preprocessing
    preprocess_data(raw_path, cleaned_path)
