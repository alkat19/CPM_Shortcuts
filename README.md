# CPM_Shortcuts

**Clinical Prediction Modeling Shortcuts** - R functions for automating clinical prediction model development and validation workflows.

## Overview

This repository provides production-ready R functions that streamline common clinical prediction modeling tasks, with a focus on robust validation strategies for multi-center studies.

### Featured Function: `iecv_modelling()`

Internal-External Cross-Validation (IECV) implementation supporting multiple model types:

| Model | Engine | Description |
|-------|--------|-------------|
| `logistic` | glm | Standard logistic regression with interpretable coefficients |
| `xgboost` | xgboost | Gradient boosted trees for potentially higher discrimination |
| `lightgbm` | lightgbm | Fast gradient boosted trees |

## Installation

```r
# Install required packages
install.packages(c(
  "tidyverse", "tidymodels", "furrr", "probably",
  "dcurves", "bonsai", "shapviz", "cli", "gridExtra"
))

# For XGBoost and LightGBM
install.packages(c("xgboost", "lightgbm"))
```

## Quick Start

```r
# Load the function
source("R/iecv_modelling.R")

# Load sample data
data <- read_csv("data/simulated_patient_data.csv")

# Run IECV with logistic regression
result_lr <- iecv_modelling(
  data = data,
  outcome = "outcome",
  predictors = c("age", "sex", "biomarker", "comorbidity"),
  cluster = "center",
  model = "logistic"
)

# View results
print(result_lr)
summary(result_lr)

# Generate plots
plot(result_lr)                        # Forest plots of all metrics
plot(result_lr, type = "calibration")  # Calibration curve
plot(result_lr, type = "dca")          # Decision curve analysis
```

## What is Internal-External Cross-Validation?

IECV is a validation strategy specifically designed for prediction models developed using **multi-center or multi-study data**. Instead of random cross-validation splits, IECV:

1. **Trains** the model on all centers except one
2. **Validates** on the held-out center (treating it as "external")
3. **Repeats** for each center, so every center serves as external validation once

This approach provides more realistic estimates of how well your model will perform when applied to **new centers** not used in model development.

```
Center A  Center B  Center C  Center D  Center E  Center F
   |         |         |         |         |         |
   v         v         v         v         v         v
[TRAIN]   [TRAIN]   [TRAIN]   [TRAIN]   [TRAIN]   [TEST]  <- Fold 1
[TRAIN]   [TRAIN]   [TRAIN]   [TRAIN]   [TEST]    [TRAIN] <- Fold 2
[TRAIN]   [TRAIN]   [TRAIN]   [TEST]    [TRAIN]   [TRAIN] <- Fold 3
   ...
```

## Features

### Multiple Metrics with Bootstrap CIs

```r
# Available metrics
metrics = c("auc", "brier", "cal_intercept", "cal_slope")

# Interpretation
# AUC > 0.7           Good discrimination
# Brier < 0.25        Good overall accuracy
# Cal Intercept ~ 0   No systematic bias
# Cal Slope ~ 1       No overfitting/underfitting
```

### Visualization Methods

```r
# Forest plots showing per-center performance
plot(result)
plot(result, type = "auc")

# Calibration plot (pooled out-of-fold predictions)
plot(result, type = "calibration")

# Decision curve analysis for clinical utility
plot(result, type = "dca")

# SHAP plots for tree models
plot(result_xgb, type = "shap")
```

### Variable Importance

```r
# Logistic regression: odds ratios with CIs
variable_importance(result_lr)

# Tree models: SHAP-based importance (default)
variable_importance(result_xgb)

# Tree models: native importance (Gain)
variable_importance(result_xgb, type = "native")
```

### SHAP Dependence Plots

```r
# Show how a predictor affects model predictions
plot_shap_dependence(result_xgb, feature = "age")
```

## Function Reference

### Main Function

```r
iecv_modelling(
  data,           # Data frame with outcome, predictors, cluster
  outcome,        # Name of binary outcome variable (0/1)
  predictors,     # Character vector of predictor names
  cluster,        # Name of clustering variable (e.g., "center")
  model,          # "logistic", "xgboost", or "lightgbm"
  metrics,        # Which metrics to compute
  n_boot = 50,    # Bootstrap replicates for CIs
  conf_level = 0.95,
  n_cores = NULL, # Parallel cores (NULL = auto)
  verbose = TRUE, # Show progress
  seed = 123
)
```

### Output Object

The function returns an `iecv_result` object containing:

- `cluster_results` - Per-cluster metrics with bootstrap CIs
- `summary` - Pooled summary statistics
- `predictions` - Out-of-fold predictions
- `final_model` - Fitted workflow on all data
- `resamples` - The rsample object

### Helper Functions

| Function | Description |
|----------|-------------|
| `variable_importance()` | Extract variable importance |
| `tidy_final_model()` | Get model coefficients (logistic) |
| `get_shap()` | Get shapviz object for custom SHAP plots |
| `plot_shap_dependence()` | SHAP dependence plot for a feature |
| `dca_table()` | Decision curve analysis table |
| `get_dca()` | Get raw dcurves DCA object |
| `format_ci()` | Format estimate with confidence interval |

## Example Dataset

The included `simulated_patient_data.csv` contains 1,346 patients across 6 hospitals:

| Column | Description |
|--------|-------------|
| patient_id | Unique identifier |
| center | Hospital (A-F) |
| age | Patient age |
| sex | Binary (0/1) |
| biomarker | Continuous value |
| comorbidity | Binary (0/1) |
| outcome | Binary outcome (0/1) |

## Running Tests

```r
# Run the test suite
testthat::test_file("tests/test-iecv_modelling.R")
```

## Demo

See `demo/iecv_demo.qmd` for an interactive tutorial with:

- Step-by-step IECV workflow
- Comparison of all three model types
- Publication-quality figures
- Interpretation guidance

## References

- Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal, internal-external, and external validation. *J Clin Epidemiol*. 2016;69:245-247.
- Austin PC, et al. Geographic and temporal validity of prediction models: different approaches were useful to examine model performance. *J Clin Epidemiol*. 2016;79:76-85.
- Debray TPA, et al. A new framework to enhance the interpretation of external validation studies of clinical prediction models. *J Clin Epidemiol*. 2015;68:279-289.

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
