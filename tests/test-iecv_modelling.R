# =============================================================================
# Unit Tests for iecv_modelling.R
# =============================================================================
#
# Run tests with: testthat::test_file("test-iecv_modelling.R")
# Or source this file and run: run_all_tests()
#
# =============================================================================

library(testthat)
library(tidyverse)
library(tidymodels)
library(probably)
library(dcurves)

# Source the main file
source("iecv_modelling.R")

# =============================================================================
# Test Data Setup
# =============================================================================

#' Create test dataset for unit tests
#' Small dataset with known properties for fast, reproducible tests
create_test_data <- function(n_per_cluster = 100, n_clusters = 4, seed = 42) {
  set.seed(seed)

  map_dfr(seq_len(n_clusters), function(i) {
    n <- n_per_cluster
    center_effect <- (i - 2.5) * 0.2

    age <- rnorm(n, mean = 60, sd = 10)
    sex <- rbinom(n, 1, 0.5)
    biomarker <- rnorm(n, mean = 5, sd = 2)
    category <- sample(c("A", "B", "C"), n, replace = TRUE)

    lp <- -2 + center_effect + 0.03 * age + 0.4 * sex + 0.2 * biomarker
    outcome <- rbinom(n, 1, plogis(lp))

    tibble(
      cluster = paste0("Center_", LETTERS[i]),
      age = age,
      sex = sex,
      biomarker = biomarker,
      category = factor(category),
      outcome = outcome
    )
  })
}

# Create test data once for all tests
test_data <- create_test_data()

# =============================================================================
# Input Validation Tests
# =============================================================================

test_that("iecv_modelling validates data argument", {
  expect_error(
    iecv_modelling(
      data = "not_a_dataframe",
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster"
    ),
    "must be a data frame"
  )
})

test_that("iecv_modelling validates outcome variable exists", {
  expect_error(
    iecv_modelling(
      data = test_data,
      outcome = "nonexistent",
      predictors = "age",
      cluster = "cluster"
    ),
    "not found in data"
  )
})

test_that("iecv_modelling validates predictor variables exist", {
  expect_error(
    iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "nonexistent"),
      cluster = "cluster"
    ),
    "not found in data"
  )
})

test_that("iecv_modelling validates cluster variable exists", {
  expect_error(
    iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "nonexistent"
    ),
    "not found in data"
  )
})

test_that("iecv_modelling validates model argument", {
  expect_error(
    iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster",
      model = "invalid_model"
    ),
    "'arg' should be one of"
  )
})

test_that("iecv_modelling validates metrics argument", {
  expect_error(
    iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster",
      metrics = c("auc", "invalid_metric")
    ),
    "Invalid metric"
  )
})

test_that("iecv_modelling requires binary outcome", {
  bad_data <- test_data %>%
    mutate(outcome = outcome + 1)

  expect_error(
    iecv_modelling(
      data = bad_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster"
    ),
    "binary"
  )
})

test_that("iecv_modelling requires at least 3 clusters", {
  small_data <- test_data %>%
    filter(cluster %in% c("Center_A", "Center_B"))

  expect_error(
    iecv_modelling(
      data = small_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster"
    ),
    "At least 3 clusters"
  )
})

test_that("iecv_modelling warns for small n_boot", {
  expect_warning(
    iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster",
      n_boot = 5
    ),
    "unstable"
  )
})

# =============================================================================
# Model Type Tests - Logistic Regression
# =============================================================================

test_that("iecv_modelling works with logistic regression (default)", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "logistic",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
  expect_equal(result$model_type, "logistic")
})

test_that("logistic model has interpretable coefficients", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "logistic",
    n_boot = 20,
    verbose = FALSE
  )

  coefs <- tidy_final_model(result)
  expect_s3_class(coefs, "tbl_df")
  expect_true("term" %in% names(coefs))
  expect_equal(nrow(coefs), 3)  # intercept + 2 predictors
})

# =============================================================================
# Model Type Tests - XGBoost
# =============================================================================

test_that("iecv_modelling works with xgboost", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
  expect_equal(result$model_type, "xgboost")
})

test_that("xgboost handles factor predictors", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "category"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
  # Should complete without error - parsnip handles factor encoding
})

test_that("xgboost predictions are valid probabilities", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  probs <- result$predictions$predicted_prob
  expect_true(all(probs >= 0 & probs <= 1))
})

# =============================================================================
# Model Type Tests - LightGBM
# =============================================================================

test_that("iecv_modelling works with lightgbm", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
  expect_equal(result$model_type, "lightgbm")
})

test_that("lightgbm handles factor predictors", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "category"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
})

test_that("lightgbm predictions are valid probabilities", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  probs <- result$predictions$predicted_prob
  expect_true(all(probs >= 0 & probs <= 1))
})

# =============================================================================
# Basic Functionality Tests
# =============================================================================

test_that("iecv_modelling returns correct class", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    n_boot = 20,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
})

test_that("iecv_modelling returns all expected components", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    n_boot = 20,
    verbose = FALSE
  )

  expected_components <- c(
    "cluster_results", "summary", "predictions", "final_model",
    "resamples", "formula", "outcome", "predictors", "cluster",
    "model_type", "metrics", "conf_level", "n_boot", "n_clusters",
    "predictor_data"
  )

  expect_true(all(expected_components %in% names(result)))
})

test_that("iecv_modelling produces correct number of folds", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = "age",
    cluster = "cluster",
    n_boot = 20,
    verbose = FALSE
  )

  n_clusters <- length(unique(test_data$cluster))
  expect_equal(nrow(result$cluster_results), n_clusters)
})

test_that("iecv_modelling stores correct metadata", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 30,
    conf_level = 0.90,
    verbose = FALSE
  )

  expect_equal(result$n_boot, 30)
  expect_equal(result$conf_level, 0.90)
  expect_equal(result$outcome, "outcome")
  expect_equal(result$predictors, c("age", "sex"))
  expect_equal(result$cluster, "cluster")
  expect_equal(result$model_type, "xgboost")
})

test_that("iecv_modelling accepts n_cores parameter", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    n_boot = 20,
    n_cores = 1,
    verbose = FALSE
  )

  expect_s3_class(result, "iecv_result")
})

test_that("iecv_modelling verbose parameter works", {
  # Test that verbose = TRUE runs without error (cli output goes to stderr)
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age"),
    cluster = "cluster",
    n_boot = 10,
    verbose = TRUE
  )
  expect_s3_class(result, "iecv_result")
})

# =============================================================================
# Metrics Computation Tests
# =============================================================================

test_that("iecv_modelling computes all default metrics", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    n_boot = 20,
    verbose = FALSE
  )

  expect_true("auc" %in% names(result$cluster_results))
  expect_true("brier" %in% names(result$cluster_results))
  expect_true("cal_intercept" %in% names(result$cluster_results))
  expect_true("cal_slope" %in% names(result$cluster_results))
  expect_true("auc_lower" %in% names(result$cluster_results))
  expect_true("auc_upper" %in% names(result$cluster_results))
})

test_that("AUC values are within valid range for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      metrics = "auc",
      n_boot = 20,
      verbose = FALSE
    )

    auc_values <- result$cluster_results$auc
    expect_true(all(auc_values >= 0 & auc_values <= 1))
  }
})

test_that("Brier scores are within valid range for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      metrics = "brier",
      n_boot = 20,
      verbose = FALSE
    )

    brier_values <- result$cluster_results$brier
    expect_true(all(brier_values >= 0 & brier_values <= 1))
  }
})

# =============================================================================
# S3 Methods Tests
# =============================================================================

test_that("print method shows model type", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = "age",
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  expect_output(print(result), "XGBoost")
  expect_output(print(result), "Internal-External Cross-Validation")
})

test_that("summary method shows model type", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = "age",
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  expect_output(summary(result), "LightGBM")
  expect_output(summary(result), "IECV Summary")
})

test_that("plot method returns ggplot objects for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      verbose = FALSE
    )

    p <- plot(result, type = "auc")
    expect_s3_class(p, "ggplot")
  }
})

test_that("calibration plot works for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      verbose = FALSE
    )

    p_cal <- plot(result, type = "calibration")
    expect_s3_class(p_cal, "ggplot")
  }
})

test_that("SHAP plot type is only valid for tree models", {
  result_logistic <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "logistic",
    n_boot = 20,
    verbose = FALSE
  )

  expect_error(
    plot(result_logistic, type = "shap"),
    "Invalid plot type"
  )
})

# =============================================================================
# Variable Importance Tests
# =============================================================================

test_that("variable_importance works for logistic regression", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "logistic",
    n_boot = 20,
    verbose = FALSE
  )

  vi <- variable_importance(result)
  expect_s3_class(vi, "tbl_df")
  expect_true("variable" %in% names(vi))
  expect_true("odds_ratio" %in% names(vi))
})

test_that("variable_importance works for xgboost (native)", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  vi <- variable_importance(result, type = "native")
  expect_s3_class(vi, "tbl_df")
  expect_true("variable" %in% names(vi))
  expect_true("importance" %in% names(vi))
})

test_that("variable_importance works for lightgbm (native)", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  vi <- variable_importance(result, type = "native")
  expect_s3_class(vi, "tbl_df")
  expect_true("variable" %in% names(vi))
  expect_true("importance" %in% names(vi))
})

test_that("variable_importance validates input", {
  expect_error(
    variable_importance("not_an_iecv_result"),
    "must be an iecv_result object"
  )
})

# =============================================================================
# SHAP Functions Tests (requires shapviz package)
# =============================================================================

test_that("get_shap validates model type", {
  result_logistic <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "logistic",
    n_boot = 20,
    verbose = FALSE
  )

  expect_error(
    get_shap(result_logistic),
    "tree-based models"
  )
})

test_that("get_shap validates input", {
  expect_error(
    get_shap("not_an_iecv_result"),
    "must be an iecv_result object"
  )
})

# Conditional SHAP tests - only run if shapviz is available
skip_if_no_shapviz <- function() {
  if (!requireNamespace("shapviz", quietly = TRUE)) {
    skip("shapviz package not available")
  }
}

test_that("get_shap works for xgboost when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  shp <- get_shap(result, n_samples = 50)
  expect_s3_class(shp, "shapviz")
})

test_that("SHAP plot works for xgboost when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  p <- plot(result, type = "shap", n_samples = 50)
  expect_s3_class(p, "ggplot")
})

test_that("variable_importance SHAP works for xgboost when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  vi <- variable_importance(result, type = "shap", n_samples = 50)
  expect_s3_class(vi, "tbl_df")
  expect_true("variable" %in% names(vi))
})

# Conditional SHAP tests for lightgbm - uses native TreeSHAP like xgboost
test_that("get_shap works for lightgbm when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  shp <- get_shap(result, n_samples = 30)
  expect_s3_class(shp, "shapviz")
})

test_that("SHAP plot works for lightgbm when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  p <- plot(result, type = "shap", n_samples = 30)
  expect_s3_class(p, "ggplot")
})

test_that("variable_importance SHAP works for lightgbm when shapviz available", {
  skip_if_no_shapviz()

  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex", "biomarker"),
    cluster = "cluster",
    model = "lightgbm",
    n_boot = 20,
    verbose = FALSE
  )

  vi <- variable_importance(result, type = "shap", n_samples = 30)
  expect_s3_class(vi, "tbl_df")
  expect_true("variable" %in% names(vi))
})

# =============================================================================
# DCA Functions Tests
# =============================================================================

test_that("dca_table works for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      verbose = FALSE
    )

    dca_tbl <- dca_table(result)
    expect_s3_class(dca_tbl, "tbl_df")
    expect_true("net_benefit" %in% names(dca_tbl))
  }
})

test_that("dca_table labels show model type", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  dca_tbl <- dca_table(result)
  labels <- unique(dca_tbl$label)

  expect_true(any(grepl("XGBoost", labels)))
})

test_that("get_dca returns dca object for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      verbose = FALSE
    )

    dca_obj <- get_dca(result)
    expect_s3_class(dca_obj, "dca")
  }
})

# =============================================================================
# tidy_final_model Tests
# =============================================================================

test_that("tidy_final_model gives message for tree models", {
  result <- iecv_modelling(
    data = test_data,
    outcome = "outcome",
    predictors = c("age", "sex"),
    cluster = "cluster",
    model = "xgboost",
    n_boot = 20,
    verbose = FALSE
  )

  expect_message(
    tidy_final_model(result),
    "variable_importance"
  )
})

test_that("tidy_final_model validates input", {
  expect_error(
    tidy_final_model("not_an_iecv_result"),
    "must be an iecv_result object"
  )
})

# =============================================================================
# Final Model Tests
# =============================================================================

test_that("Final model can make predictions for all model types", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = c("age", "sex"),
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      verbose = FALSE
    )

    new_data <- tibble(age = c(50, 70), sex = c(0, 1))
    preds <- predict(result$final_model, new_data, type = "prob")

    expect_equal(nrow(preds), 2)
    expect_true(".pred_yes" %in% names(preds))
  }
})

# =============================================================================
# Reproducibility Tests
# =============================================================================

test_that("iecv_modelling is reproducible with seed for all models", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    result1 <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      seed = 123,
      verbose = FALSE
    )

    result2 <- iecv_modelling(
      data = test_data,
      outcome = "outcome",
      predictors = "age",
      cluster = "cluster",
      model = model_type,
      n_boot = 20,
      seed = 123,
      verbose = FALSE
    )

    expect_equal(
      result1$cluster_results$auc,
      result2$cluster_results$auc
    )
  }
})

# =============================================================================
# Model Specification Helper Tests
# =============================================================================

test_that("get_model_spec returns valid parsnip specs", {
  for (model_type in c("logistic", "xgboost", "lightgbm")) {
    spec <- get_model_spec(model_type)
    expect_s3_class(spec, "model_spec")
  }
})

test_that("get_model_spec errors on invalid model", {
  expect_error(
    get_model_spec("invalid_model"),
    "Unknown model type"
  )
})

test_that("get_model_label returns correct labels", {
  expect_equal(get_model_label("logistic"), "Logistic Regression")
  expect_equal(get_model_label("xgboost"), "XGBoost")
  expect_equal(get_model_label("lightgbm"), "LightGBM")
})

# =============================================================================
# Utility Function Tests
# =============================================================================

test_that("format_ci produces correct output", {
  result <- format_ci(0.75, 0.70, 0.80)
  expect_equal(result, "0.750 (0.700-0.800)")

  result_2digits <- format_ci(0.75, 0.70, 0.80, digits = 2)
  expect_equal(result_2digits, "0.75 (0.70-0.80)")
})

# =============================================================================
# Run All Tests
# =============================================================================

#' Run all tests and report results
#' @export
run_all_tests <- function() {
  cat("Running IECV modelling unit tests...\n\n")

  test_results <- testthat::test_file(
    "test-iecv_modelling.R",
    reporter = "summary"
  )

  cat("\n")
  invisible(test_results)
}

# Run tests if sourced directly
if (sys.nframe() == 0) {
  run_all_tests()
}
