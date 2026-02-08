import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_data(df, is_training=True):
    """
    Preprocess bike sharing dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset with features and target
    is_training : bool, default=True
        Whether this is training data (includes target variable)
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series (if is_training=True)
        Target vector
    """
    df_processed = df.copy()
    
    # Drop the date column (not numeric, already have month/weekday)
    if 'dteday' in df_processed.columns:
        df_processed = df_processed.drop('dteday', axis=1)
    
    # Separate features and target
    if is_training and 'cnt' in df_processed.columns:
        X = df_processed.drop('cnt', axis=1)
        y = df_processed['cnt']
        return X, y
    else:
        return df_processed


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a regression model, returning comprehensive metrics.
    
    Parameters:
    -----------
    model : sklearn estimator
        Untrained model object with fit/predict methods
    X_train, X_test : pandas.DataFrame
        Training and test feature matrices
    y_train, y_test : pandas.Series
        Training and test target vectors
    
    Returns:
    --------
    metrics : dict
        Dictionary containing train and test metrics:
        - train_rmse, test_rmse
        - train_mae, test_mae
        - train_r2, test_r2
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for both train and test sets
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return metrics


def calculate_drift_statistics(df1, df2, features=None):
    """
    Calculate drift statistics between two datasets.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        Two datasets to compare
    features : list, optional
        List of feature names to analyze. If None, uses all numeric columns.
    
    Returns:
    --------
    drift_stats : pandas.DataFrame
        DataFrame with drift statistics for each feature:
        - mean_1, mean_2, mean_change_%
        - std_1, std_2, std_change_%
    """
    if features is None:
        features = df1.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_1 = df1[features].describe().T[['mean', 'std']]
    stats_1.columns = ['mean_1', 'std_1']
    
    stats_2 = df2[features].describe().T[['mean', 'std']]
    stats_2.columns = ['mean_2', 'std_2']
    
    drift_stats = pd.concat([stats_1, stats_2], axis=1)
    
    # Calculate percentage changes
    drift_stats['mean_change_%'] = (
        (drift_stats['mean_2'] - drift_stats['mean_1']) / 
        drift_stats['mean_1'] * 100
    )
    drift_stats['std_change_%'] = (
        (drift_stats['std_2'] - drift_stats['std_1']) / 
        drift_stats['std_1'] * 100
    )
    
    return drift_stats


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract and sort feature importance from a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=10
        Number of top features to return
    
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with features and their importance scores, sorted descending
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def calculate_prediction_intervals(model, X, confidence=0.95, n_bootstraps=100):
    """
    Calculate prediction intervals using bootstrap resampling.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pandas.DataFrame
        Features to predict on
    confidence : float, default=0.95
        Confidence level for intervals
    n_bootstraps : int, default=100
        Number of bootstrap samples
    
    Returns:
    --------
    intervals : dict
        Dictionary with 'lower', 'upper', and 'mean' predictions
    """
    predictions = []
    
    # Generate bootstrap predictions
    for _ in range(n_bootstraps):
        # Bootstrap sample (with replacement)
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.iloc[indices]
        
        # Predict
        y_pred = model.predict(X_boot)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    intervals = {
        'mean': np.mean(predictions, axis=0),
        'lower': np.percentile(predictions, lower_percentile, axis=0),
        'upper': np.percentile(predictions, upper_percentile, axis=0)
    }
    
    return intervals


def print_model_summary(model, metrics, model_name="Model"):
    """
    Print a formatted summary of model performance.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    metrics : dict
        Dictionary of evaluation metrics
    model_name : str, default="Model"
        Name of the model for display
    """
    print("\n" + "="*80)
    print(f"{model_name} - Performance Summary")
    print("="*80)
    
    print("\nTraining Set:")
    print(f"  RMSE: {metrics['train_rmse']:.2f}")
    print(f"  MAE:  {metrics['train_mae']:.2f}")
    print(f"  R²:   {metrics['train_r2']:.4f}")
    
    print("\nTest Set:")
    print(f"  RMSE: {metrics['test_rmse']:.2f}")
    print(f"  MAE:  {metrics['test_mae']:.2f}")
    print(f"  R²:   {metrics['test_r2']:.4f}")
    
    print("\nGeneralization:")
    rmse_diff = metrics['train_rmse'] - metrics['test_rmse']
    r2_diff = metrics['train_r2'] - metrics['test_r2']
    print(f"  RMSE Gap:  {rmse_diff:+.2f}")
    print(f"  R² Gap:    {r2_diff:+.4f}")
    
    if r2_diff < 0.05:
        print("  Status:    Good generalization")
    elif r2_diff < 0.10:
        print("  Status:    Acceptable generalization")
    else:
        print("  Status:    Possible overfitting")
    
    print("="*80)


# Constants
FEATURE_NAMES = [
    'season', 'mnth', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]

SEASON_MAPPING = {
    1: 'Spring',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
}

WEATHER_MAPPING = {
    1: 'Clear',
    2: 'Mist/Cloudy',
    3: 'Light Rain/Snow',
    4: 'Heavy Rain/Snow'
}


if __name__ == "__main__":
    # Simple test
    print("Utils module loaded successfully!")
    print(f"Available functions: {[name for name in dir() if not name.startswith('_')]}")
