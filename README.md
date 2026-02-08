# ML2_ASG_GitHub_Alluru-Rishitha-Reddy
Project Overview

This project implements a complete MLOps pipeline for predicting daily bike sharing demand using the Capital Bikeshare dataset from Washington D.C. The project demonstrates best practices in:

- Systematic experiment design and model development
- ML experiment tracking with MLflow
- Data drift detection and analysis
- Automated CI/CD with GitHub Actions
- Model quality gates and validation

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Data Drift Analysis](#data-drift-analysis)
- [CI/CD Pipeline](#cicd-pipeline)
- [Quality Gates](#quality-gates)

## Project Structure

```
bike-sharing-mlops/
├── .github/
│   └── workflows/
│       └── python-app.yml          # GitHub Actions workflow 
├── src/                             # For Modular Code
│   ├── train_model.py              # Standalone training script
│   └── utils.py                    # Utility functions
├── tests/
│   └── test_model.py               # Quality gate tests
├── data/
│   ├── day_2011.csv                # Training data
│   └── day_2012.csv                # Drift analysis data
├── models/
│   └── best_model.joblib           # Saved model
├── notebooks/
│   └── ML2_ASG_notebook.ipynb      # Main analysis notebook
├── requirements.txt                 # Python dependencies 
├── baseline_rmse.txt               # Baseline performance metric
└── README.md                       # This file
```

**Note:** The `src/` folder with `train_model.py` and `utils.py` is **OPTIONAL**. 
These files are provided for those who want to run training from command line 
or create a more modular code structure. For the assignment, all required code 
is in the Jupyter notebook.

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd bike-sharing-mlops
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import sklearn, mlflow, pandas; print('All packages installed successfully!')"
```

## Usage

### Running the Jupyter Notebook (Primary Method)

The main analysis is contained in the Jupyter notebook:

```bash
jupyter notebook notebooks/ML2_ASG_Alluru Rishitha Reddy.ipynb
```

The notebook contains:
- Complete data preprocessing pipeline
- Three experimental model developments (Linear Regression, Ridge, XGBoost)
- MLflow experiment tracking
- Data drift analysis between 2011 and 2012
- Comprehensive visualizations and insights

**This notebook contains everything required for the assignment.**

### Running the Model Training Script (Optional Alternative)

For those who prefer command-line execution or want to integrate training 
into automated pipelines, a standalone training script is provided:

```bash
python src/train_model.py --data data/day_2011.csv --output models/
```

This will:
- Load and preprocess the data
- Train all three models (Linear Regression, Ridge, XGBoost)
- Log experiments to MLflow
- Save the best model to `models/best_model.joblib`

**Note:** This is optional and not required for the assignment. The notebook 
is sufficient.

### Running Tests

Execute the quality gate tests:

```bash
python tests/test_model.py
```

This will:
- Load the saved model
- Evaluate performance on test data
- Check if RMSE meets quality threshold
- Exit with status code 0 (pass) or 1 (fail)

### Viewing MLflow Experiments

Start the MLflow UI to view all tracked experiments:

```bash
mlflow ui
```

Then open your browser to `http://localhost:5000` to explore:
- All experiment runs
- Model parameters and metrics
- Comparison between different models
- Registered models

## Model Development

### Experiment Design

We follow a systematic approach to model development:

#### Experiment 1: Baseline - Linear Regression
- **Purpose**: Establish simple, interpretable baseline
- **Rationale**: Understand basic linear relationships
- **Expected**: Moderate performance, potential underfitting

#### Experiment 2: Ridge Regression
- **Purpose**: Improve generalization with L2 regularization
- **Rationale**: Handle multicollinearity, prevent overfitting
- **Expected**: Better generalization than baseline

#### Experiment 3: XGBoost (with constraints)
- **Purpose**: Capture non-linear patterns while controlling complexity
- **Rationale**: Balance performance with interpretability
- **Expected**: Best performance with controlled overfitting

### Model Selection Criteria

Models are evaluated and selected based on:
1. **Performance**: Test RMSE, MAE, R² score
2. **Generalization**: Train-test performance gap
3. **Robustness**: Expected stability on new data
4. **Complexity**: Maintainability and interpretability

### Key Features

The model uses the following features:
- **season**: Season (1=spring, 2=summer, 3=fall, 4=winter)
- **mnth**: Month (1-12)
- **holiday**: Whether day is holiday
- **weekday**: Day of week
- **workingday**: Working day indicator
- **weathersit**: Weather situation (1-4)
- **temp**: Normalized temperature
- **atemp**: Normalized feeling temperature
- **hum**: Normalized humidity
- **windspeed**: Normalized wind speed

## Data Drift Analysis

### Approach

We analyze temporal data drift by:
1. Comparing feature distributions between 2011 and 2012
2. Calculating descriptive statistics (mean, std dev)
3. Visualizing distribution shifts
4. Assessing impact on model performance

### Key Findings

- **Target Variable**: Significant growth in bike sharing usage from 2011 to 2012
- **Feature Drift**: Several features show notable distribution changes
- **Performance Impact**: Model performance degrades when applied to 2012 data
- **Contributors**: High-importance features with significant drift contribute most to degradation

## CI/CD Pipeline

### GitHub Actions Workflow

The project uses GitHub Actions for automated testing:

**Triggers**:
- Push to `main` branch
- Pull requests to `main` branch

**Steps**:
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Run linting checks
5. Execute quality gate tests

### Workflow Configuration

Located at `.github/workflows/python-app.yml`:
- Uses Ubuntu latest runner
- Python 3.10 environment
- Automated dependency installation
- Flake8 for code quality
- Custom test suite execution

## Quality Gates

### Implementation

The quality gate ensures model performance meets standards before deployment:

```python
# Quality Gate Logic
baseline_rmse = 700.0  # From baseline model
threshold = 0.95 * baseline_rmse  # 95% of baseline

assert model_rmse <= threshold, "Model performance below acceptable threshold"
```

### Benefits

1. **Prevent Regression**: Stops deployment of inferior models
2. **Automated Validation**: No manual checks required
3. **Immediate Feedback**: Fast notification of issues
4. **Version Control**: Only passing models reach production
5. **Audit Trail**: Complete testing history in GitHub Actions

### How It Protects Production

- Validates every model change
- Ensures consistent quality standards
- Prevents accidental deployment of poorly performing models
- Maintains production system reliability
- Supports continuous improvement without risk

## MLOps Best Practices Demonstrated

✅ **Experiment Tracking**: Comprehensive MLflow logging  
✅ **Version Control**: Git for code and model versioning  
✅ **Automated Testing**: CI/CD with GitHub Actions  
✅ **Quality Assurance**: Automated quality gates  
✅ **Monitoring**: Drift detection and analysis  
✅ **Documentation**: Detailed README and inline comments  
✅ **Reproducibility**: Fixed random seeds and environment specs  

## Future Improvements

### Model Enhancements
- Feature engineering (interaction terms)
- Time series models (ARIMA, Prophet)
- Ensemble methods
- Hyperparameter optimization (Bayesian, GridSearch)

### Drift Handling
- Online learning implementation
- Seasonal model variants
- Automated retraining triggers
- Drift detection alerts

### Automation
- Performance monitoring dashboard
- Automated data quality checks
- A/B testing framework
- Stakeholder reporting automation
- Model versioning and rollback

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project was developed as part of the Machine Learning 2 module at Ngee Ann Polytechnic's Diploma in Data Science program.
