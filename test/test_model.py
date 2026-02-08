import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sys
import os


def load_data(filepath='data/day_2011.csv'):
    df = pd.read_csv(filepath)
    
    # Drop date column
    if 'dteday' in df.columns:
        df = df.drop('dteday', axis=1)
    
    # Separate features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']
    
    return X, y


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'mae': mean_absolute_error(y, predictions),
        'r2': r2_score(y, predictions)
    }
    
    return metrics


def test_model_performance():
    """
    Main test function that implements the quality gate.
    
    Quality Gate Criteria:
    - RMSE must be <= 95% of baseline RMSE
    - This ensures the model performs better than or similar to baseline
    """
    print("=" * 80)
    print("Automated Model Quality Gate Test")
    
    # Load the model
    try:
        model = joblib.load('models/best_model.joblib')
        print("Model loaded successfully")
    except FileNotFoundError:
        print("ERROR: Model file not found at 'models/best_model.joblib'")
        sys.exit(1)
    
    # Load and prepare data
    try:
        X, y = load_data()
        print(f"Data loaded successfully: {X.shape[0]} samples")
    except FileNotFoundError:
        print("ERROR: Data file not found")
        sys.exit(1)
    
    # Split data (same split as training for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Evaluate model on test set
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
    
    # Quality Gate: RMSE Threshold
    # The model must perform at least as well as 95% of baseline
    # Baseline RMSE is approximately 700-800 for Linear Regression
    
    # Load baseline RMSE if available
    try:
        with open('baseline_rmse.txt', 'r') as f:
            baseline_rmse = float(f.read().strip())
    except FileNotFoundError:
        # If baseline not found, use conservative estimate
        baseline_rmse = 800.0
        print(f"\nWarning: Using estimated baseline RMSE: {baseline_rmse:.2f}")
    
    threshold = 0.95 * baseline_rmse
    
    print("\n" + "=" * 80)
    print("Quality Gate Evaluation")
    print(f"Baseline RMSE:    {baseline_rmse:.2f}")
    print(f"Threshold (95%):  {threshold:.2f}")
    print(f"Model RMSE:       {metrics['rmse']:.2f}")
    
    # Check if model passes quality gate
    if metrics['rmse'] <= threshold:
        print(f"\nPASS: Model RMSE ({metrics['rmse']:.2f}) is within acceptable threshold ({threshold:.2f})")
        print("Quality gate passed - Model is ready for deployment")
        return True
    else:
        print(f"\nFAIL: Model RMSE ({metrics['rmse']:.2f}) exceeds threshold ({threshold:.2f})")
        print("Quality gate failed - Model performance is below acceptable standards")
        print("\nRecommendation: Review model training, hyperparameters, or data quality")
        sys.exit(1)


def test_model_exists():
    """Test that the model file exists."""
    assert os.path.exists('models/best_model.joblib'), "Model file not found"
    print("Test passed: Model file exists")


def test_data_exists():
    """Test that the data file exists."""
    assert os.path.exists('data/day_2011.csv'), "Data file not found"
    print("Test passed: Data file exists")


if __name__ == "__main__":
    print("Running Model Tests")
    
    # Run basic tests
    test_model_exists()
    test_data_exists()
    
    # Run main quality gate test
    test_model_performance()
    
    print("All tests passed!!")