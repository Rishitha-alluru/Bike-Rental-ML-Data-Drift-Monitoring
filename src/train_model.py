"""
Train bike sharing demand prediction models and log results to MLflow.

Usage Path:
    python src/train_model.py --data data/day_2011.csv --output models/
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import preprocess_data, evaluate_model

def train_baseline_model(X_train, X_test, y_train, y_test):
    #Train baseline Linear Regression model.
    print("Training Baseline Model: Linear Regression")
    
    with mlflow.start_run(run_name="Baseline_Linear_Regression"):
        model = LinearRegression()
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Log to MLflow
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_metric("test_rmse", metrics['test_rmse'])
        mlflow.log_metric("test_mae", metrics['test_mae'])
        mlflow.log_metric("test_r2", metrics['test_r2'])
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        
        return model, metrics


def train_ridge_model(X_train, X_test, y_train, y_test):
    #Train Ridge Regression model.
    print("Training Improved Model: Ridge Regression")
    
    with mlflow.start_run(run_name="Improved_Ridge_Regression"):
        model = Ridge(alpha=10.0, random_state=42)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Log to MLflow
        mlflow.log_param("model_type", "Ridge Regression")
        mlflow.log_param("alpha", 10.0)
        mlflow.log_metric("test_rmse", metrics['test_rmse'])
        mlflow.log_metric("test_mae", metrics['test_mae'])
        mlflow.log_metric("test_r2", metrics['test_r2'])
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        
        return model, metrics


def train_xgboost_model(X_train, X_test, y_train, y_test):
    #Train XGBoost model with depth constraint
    print("Training Best Model: XGBoost")
    
    with mlflow.start_run(run_name="Best_XGBoost"):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=10,
            gamma=0.1,
            random_state=42
        )
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Log to MLflow
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("min_child_weight", 10)
        mlflow.log_param("gamma", 0.1)
        mlflow.log_metric("test_rmse", metrics['test_rmse'])
        mlflow.log_metric("test_mae", metrics['test_mae'])
        mlflow.log_metric("test_r2", metrics['test_r2'])
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        
        return model, metrics


def main(data_path, output_dir):
    #Main training pipeline.
    print("BIKE SHARING DEMAND PREDICTION - MODEL TRAINING")
    
    # Set MLflow experiment
    mlflow.set_experiment("Bike_Sharing_Demand_Prediction")
    
    # Load and preprocess data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train all models
    lr_model, lr_metrics = train_baseline_model(X_train, X_test, y_train, y_test)
    ridge_model, ridge_metrics = train_ridge_model(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics = train_xgboost_model(X_train, X_test, y_train, y_test)
    
    # Select best model (XGBoost)
    best_model = xgb_model
    best_metrics = xgb_metrics
    baseline_rmse = lr_metrics['test_rmse']

    print("\nModel selection")
    print(f"Best Model: XGBoost")
    print(f"Test RMSE: {best_metrics['test_rmse']:.2f}")
    print(f"Improvement over baseline: {((baseline_rmse - best_metrics['test_rmse']) / baseline_rmse * 100):.1f}%")
    
    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save baseline RMSE for quality gate
    baseline_path = os.path.join(output_dir, '../baseline_rmse.txt')
    with open(baseline_path, 'w') as f:
        f.write(str(baseline_rmse))
    print(f"Baseline RMSE saved to: {baseline_path}")

    print("\nTraining complete!")
    print(f"\nView experiments: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train bike sharing demand prediction model')
    parser.add_argument('--data', type=str, default='data/day_2011.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='models/',
                        help='Directory to save trained model')
    
    args = parser.parse_args()
    
    main(args.data, args.output)
