# =============================================================================
# Internal-External Cross-Validation with Multiple Model Types
# =============================================================================
#
# A tidymodels-based implementation of Internal-External Cross-Validation (IECV)
# for clinical prediction models supporting multiple model types:
# - Logistic Regression
# - XGBoost
# - LightGBM
#
# Features:
# - Uses rsample::group_vfold_cv for leave-one-cluster-out splits
# - Uses parsnip/workflows for model specification
# - Uses yardstick for standard metrics
# - Custom calibration metrics (slope, intercept)
# - Bootstrap confidence intervals per cluster
# - SHAP plots for tree-based models (shapviz)
# - S3 class with plot methods
# - Verbose progress output
# - Configurable parallel processing
#
# Dependencies: tidyverse, tidymodels, furrr, probably, dcurves, shapviz, bonsai
# =============================================================================

library(tidyverse)
library(tidymodels)
library(furrr)
library(probably)
library(dcurves)
library(bonsai)  # For LightGBM support via parsnip
library(cli)     # For verbose output

# =============================================================================
# Model Specification Helpers
# =============================================================================

#' Get model specification based on model type
#'
#' @param model Character string: "logistic", "xgboost", or "lightgbm"
#' @return A parsnip model specification
#' @keywords internal
get_model_spec <- function(model) {
  switch(
    model,
    logistic = parsnip::logistic_reg() %>%
      parsnip::set_engine("glm") %>%
      parsnip::set_mode("classification"),

    xgboost = parsnip::boost_tree(
      trees = 100,
      tree_depth = 6,
      learn_rate = 0.1,
      min_n = 10,
      loss_reduction = 0,
      sample_size = 0.8
    ) %>%
      parsnip::set_engine("xgboost") %>%
      parsnip::set_mode("classification"),

    lightgbm = parsnip::boost_tree(
      trees = 100,
      tree_depth = 6,
      learn_rate = 0.1,
      min_n = 10,
      loss_reduction = 0
    ) %>%
      parsnip::set_engine("lightgbm") %>%
      parsnip::set_mode("classification"),

    stop(
      sprintf("Unknown model type: '%s'. Valid options: 'logistic', 'xgboost', 'lightgbm'", model),
      call. = FALSE
    )
  )
}


#' Get display name for model type
#' @keywords internal
get_model_label <- function(model) {
  switch(
    model,
    logistic = "Logistic Regression",
    xgboost = "XGBoost",
    lightgbm = "LightGBM",
    model
  )
}


# =============================================================================
# Input Validation
# =============================================================================

#' Assert input is an iecv_result object
#' @keywords internal
assert_iecv_result <- function(x) {
  if (!inherits(x, "iecv_result")) {
    stop("Input must be an iecv_result object.", call. = FALSE)
  }
  invisible(TRUE)
}


#' Assert model is tree-based
#' @keywords internal
assert_tree_model <- function(x) {
  assert_iecv_result(x)
  if (!x$model_type %in% c("xgboost", "lightgbm")) {
    stop("This function is only available for tree-based models.", call. = FALSE)
  }
  invisible(TRUE)
}


#' Validate inputs for IECV function
#' @keywords internal
validate_iecv_inputs <- function(data, outcome, predictors, cluster, metrics, n_boot) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }

  if (!outcome %in% names(data)) {
    stop(sprintf("Outcome variable '%s' not found in data.", outcome), call. = FALSE)
  }

  missing_predictors <- setdiff(predictors, names(data))
  if (length(missing_predictors) > 0) {
    stop(
      sprintf("Predictor(s) not found in data: %s", paste(missing_predictors, collapse = ", ")),
      call. = FALSE
    )
  }

  if (!cluster %in% names(data)) {
    stop(sprintf("Cluster variable '%s' not found in data.", cluster), call. = FALSE)
  }

  valid_metrics <- c("auc", "brier", "cal_intercept", "cal_slope")
  invalid_metrics <- setdiff(metrics, valid_metrics)
  if (length(invalid_metrics) > 0) {
    stop(
      sprintf(
        "Invalid metric(s): %s. Valid options: %s",
        paste(invalid_metrics, collapse = ", "),
        paste(valid_metrics, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (n_boot < 10) {
    warning("n_boot < 10 may produce unstable confidence intervals.", call. = FALSE)
  }

  unique_outcomes <- unique(data[[outcome]])
  if (!all(unique_outcomes %in% c(0, 1, NA))) {
    stop("Outcome variable must be binary (0/1).", call. = FALSE)
  }

  n_clusters <- length(unique(data[[cluster]]))
  if (n_clusters < 3) {
    stop("At least 3 clusters are required for IECV.", call. = FALSE)
  }

  invisible(TRUE)
}


# =============================================================================
# Main IECV Function
# =============================================================================

#' Internal-External Cross-Validation for Clinical Prediction Models
#'
#' Performs internal-external cross-validation (IECV) using tidymodels with
#' support for multiple model types including logistic regression, XGBoost,
#' and LightGBM.
#'
#' @param data A data frame containing the outcome, predictors, and cluster variable.
#' @param outcome Character string naming the binary outcome variable (coded as 0/1).
#' @param predictors Character vector of predictor variable names.
#' @param cluster Character string naming the cluster variable (e.g., "center", "hospital").
#' @param model Character string specifying the model type. One of:
#'   \itemize{
#'     \item \code{"logistic"}: Logistic regression (GLM)
#'     \item \code{"xgboost"}: Gradient boosted trees (XGBoost)
#'     \item \code{"lightgbm"}: Gradient boosted trees (LightGBM)
#'   }
#'   Default is "logistic".
#' @param metrics Character vector specifying which metrics to compute.
#'   Options: "auc", "brier", "cal_intercept", "cal_slope".
#'   Default is all four metrics.
#' @param n_boot Number of bootstrap replicates for confidence interval estimation.
#'   Default is 50. Increase for more stable CIs (e.g., 200 for publication).
#' @param conf_level Confidence level for intervals. Default is 0.95 (95% CI).
#' @param n_cores Number of cores for parallel processing. Default is NULL which
#'   uses all available cores minus 1. Set to 1 for sequential processing.
#' @param verbose Logical; if TRUE (default), prints progress messages.
#' @param seed Random seed for reproducibility. Default is 123.
#'
#' @return An object of class "iecv_result" containing:
#'   \itemize{
#'     \item \code{cluster_results}: Tibble with per-cluster metrics and CIs
#'     \item \code{summary}: Summary statistics across all clusters
#'     \item \code{predictions}: Out-of-fold predictions with probabilities
#'     \item \code{final_model}: Model fitted on all data (workflow object)
#'     \item \code{resamples}: The rsample object used for splitting
#'     \item \code{formula}: The model formula
#'     \item \code{model_type}: The model type used
#'     \item \code{metrics}: Character vector of computed metrics
#'     \item \code{conf_level}: The confidence level used
#'     \item \code{n_boot}: Number of bootstrap replicates used
#'   }
#'
#' @details
#' The function implements leave-one-cluster-out cross-validation, where each
#' unique value of the cluster variable serves as an external validation set
#' while the model is trained on all other clusters.
#'
#' \strong{Model Types:}
#' \itemize{
#'   \item \code{logistic}: Standard logistic regression. Interpretable coefficients.
#'   \item \code{xgboost}: Gradient boosted trees. Often higher discrimination,
#'     uses sensible defaults (100 trees, depth 6, learning rate 0.1).
#'   \item \code{lightgbm}: LightGBM gradient boosted trees. Fast training,
#'     uses sensible defaults (100 trees, depth 6, learning rate 0.1).
#' }
#'
#' \strong{Metrics:}
#' \itemize{
#'   \item \code{auc}: Area under the ROC curve (discrimination)
#'   \item \code{brier}: Brier score (overall accuracy)
#'   \item \code{cal_intercept}: Calibration-in-the-large (systematic bias)
#'   \item \code{cal_slope}: Calibration slope (<1 = overfitting, >1 = underfitting)
#' }
#'
#' \strong{Preprocessing:}
#' Factor variables are automatically handled by parsnip using one-hot encoding
#' for tree-based models. No manual preprocessing is required.
#'
#' @references
#' Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal,
#' internal-external, and external validation. J Clin Epidemiol. 2016;69:245-247.
#'
#' @examples
#' \dontrun{
#' # Logistic regression (default)
#' results_lr <- iecv_modelling(
#'   data = patient_data,
#'   outcome = "death",
#'   predictors = c("age", "sex", "biomarker"),
#'   cluster = "hospital",
#'   model = "logistic"
#' )
#'
#' # XGBoost
#' results_xgb <- iecv_modelling(
#'   data = patient_data,
#'   outcome = "death",
#'   predictors = c("age", "sex", "biomarker"),
#'   cluster = "hospital",
#'   model = "xgboost"
#' )
#'
#' # LightGBM
#' results_lgb <- iecv_modelling(
#'   data = patient_data,
#'   outcome = "death",
#'   predictors = c("age", "sex", "biomarker"),
#'   cluster = "hospital",
#'   model = "lightgbm"
#' )
#'
#' # View results
#' print(results_xgb)
#' summary(results_xgb)
#'
#' # Plot methods
#' plot(results_xgb)                      # Forest plots
#' plot(results_xgb, type = "calibration") # Calibration curve
#' plot(results_xgb, type = "shap")        # SHAP summary (tree models only)
#'
#' # Variable importance for tree models
#' variable_importance(results_xgb)
#' }
#'
#' @export
iecv_modelling <- function(data,
                           outcome,
                           predictors,
                           cluster,
                           model = c("logistic", "xgboost", "lightgbm"),
                           metrics = c("auc", "brier", "cal_intercept", "cal_slope"),
                           n_boot = 50,
                           conf_level = 0.95,
                           n_cores = NULL,
                           verbose = TRUE,
                           seed = 123) {

  model <- match.arg(model)

  # Set up parallel processing
 if (is.null(n_cores)) {
    n_cores <- max(1, parallel::detectCores() - 1)
  }
  plan(multisession, workers = n_cores)

  if (verbose) {
    cli_h1("Internal-External Cross-Validation")
    cli_alert_info("Model: {get_model_label(model)}")
    cli_alert_info("Using {n_cores} core(s) for parallel processing")
  }

  # Input Validation
  if (verbose) cli_alert("Validating inputs...")
  validate_iecv_inputs(data, outcome, predictors, cluster, metrics, n_boot)

  n_clusters <- length(unique(data[[cluster]]))
  if (verbose) cli_alert_success("Found {n_clusters} clusters for cross-validation")

  # Data Preparation
  if (verbose) cli_alert("Preparing data...")
  data_prepared <- data %>%
    mutate(
      .outcome_factor = factor(.data[[outcome]], levels = c(0, 1), labels = c("no", "yes"))
    )

  model_formula <- as.formula(
    paste(".outcome_factor ~", paste(predictors, collapse = " + "))
  )
  if (verbose) cli_alert_info("Predictors: {paste(predictors, collapse = ', ')}")

  # Create Leave-One-Cluster-Out Resamples
  if (verbose) cli_alert("Creating leave-one-cluster-out resamples...")
  set.seed(seed)
  resamples <- rsample::group_vfold_cv(data_prepared, group = !!sym(cluster), v = NULL)

  # Define Workflow
  if (verbose) cli_alert("Building model workflow...")
  model_spec <- get_model_spec(model)
  wf <- workflows::workflow() %>%
    workflows::add_model(model_spec) %>%
    workflows::add_formula(model_formula)

  # Fit Resamples
  if (verbose) cli_alert("Fitting models across {n_clusters} folds...")
  resample_results <- tune::fit_resamples(
    wf,
    resamples = resamples,
    control = tune::control_resamples(save_pred = TRUE, verbose = FALSE)
  )
  if (verbose) cli_alert_success("Cross-validation complete")

  # Extract Predictions and Compute Metrics
  if (verbose) cli_alert("Computing metrics with {n_boot} bootstrap replicates...")
  predictions_raw <- tune::collect_predictions(resample_results)

  cluster_results <- predictions_raw %>%
    group_by(id) %>%
    group_map(~ {
      truth_numeric <- as.numeric(.x$.outcome_factor == "yes")
      compute_fold_metrics(
        truth = truth_numeric,
        probs = .x$.pred_yes,
        fold_id = .y$id,
        metrics = metrics,
        n_boot = n_boot,
        conf_level = conf_level
      )
    }) %>%
    bind_rows()
  if (verbose) cli_alert_success("Metrics computed for all clusters")

  # Fit Final Model
  if (verbose) cli_alert("Fitting final model on all data...")
  final_model <- workflows::fit(wf, data = data_prepared)
  if (verbose) cli_alert_success("Final model fitted")

  # Prepare Predictions Output
  predictions_df <- predictions_raw %>%
    transmute(
      .row,
      fold = id,
      outcome = as.numeric(.outcome_factor == "yes"),
      predicted_prob = .pred_yes,
      linear_predictor = qlogis(.pred_yes)
    ) %>%
    arrange(.row)

  # Construct Result Object
  result <- structure(
    list(
      cluster_results = cluster_results,
      summary = compute_summary_stats(cluster_results, metrics),
      predictions = predictions_df,
      final_model = final_model,
      resamples = resamples,
      formula = model_formula,
      outcome = outcome,
      predictors = predictors,
      cluster = cluster,
      model_type = model,
      metrics = metrics,
      conf_level = conf_level,
      n_boot = n_boot,
      n_clusters = n_clusters,
      predictor_data = data_prepared %>% select(all_of(predictors))
    ),
    class = c("iecv_result", "list")
  )

  if (verbose) cli_alert_success("IECV complete!")
  result
}


# =============================================================================
# Metric Computation Helpers
# =============================================================================

#' Compute point estimates for all metrics
#' @keywords internal
compute_point_estimates <- function(truth, probs, truth_factor, metrics) {
  estimates <- list()

  if ("auc" %in% metrics) {
    estimates$auc <- tryCatch(
      yardstick::roc_auc_vec(truth_factor, probs, event_level = "second"),
      error = function(e) NA_real_
    )
  }

  if ("brier" %in% metrics) {
    estimates$brier <- mean((probs - truth)^2)
  }

  needs_calibration <- any(c("cal_intercept", "cal_slope") %in% metrics)
  if (needs_calibration) {
    cal_coefs <- tryCatch({
      lp <- qlogis(probs)
      coef(glm(truth ~ lp, family = binomial))
    }, error = function(e) c(NA_real_, NA_real_))

    if ("cal_intercept" %in% metrics) {
      estimates$cal_intercept <- unname(cal_coefs[1])
    }
    if ("cal_slope" %in% metrics) {
      estimates$cal_slope <- unname(cal_coefs[2])
    }
  }

  estimates
}


#' Compute metrics with bootstrap CIs for a single validation fold
#' @keywords internal
compute_fold_metrics <- function(truth, probs, fold_id, metrics, n_boot, conf_level) {
  truth_factor <- factor(truth, levels = c(0, 1), labels = c("no", "yes"))
  point_estimates <- compute_point_estimates(truth, probs, truth_factor, metrics)
  boot_cis <- compute_bootstrap_cis(truth, probs, truth_factor, metrics, n_boot, conf_level)

  result <- tibble(
    id = fold_id,
    n_validation = length(truth),
    n_events = sum(truth)
  )

  for (m in metrics) {
    result[[m]] <- point_estimates[[m]]
    result[[paste0(m, "_lower")]] <- boot_cis[[m]]$lower
    result[[paste0(m, "_upper")]] <- boot_cis[[m]]$upper
    result[[paste0(m, "_se")]] <- boot_cis[[m]]$se
  }

  result
}


#' Compute bootstrap confidence intervals for validation metrics
#' @keywords internal
compute_bootstrap_cis <- function(truth, probs, truth_factor, metrics, n_boot, conf_level) {
  alpha <- 1 - conf_level
  n_obs <- length(truth)

  boot_one <- function(i) {
    idx <- sample(n_obs, replace = TRUE)
    est <- compute_point_estimates(truth[idx], probs[idx], truth_factor[idx], metrics)
    unlist(est[metrics])
  }

  boot_results <- furrr::future_map(
    seq_len(n_boot),
    boot_one,
    .options = furrr::furrr_options(seed = TRUE)
  )
  boot_matrix <- do.call(rbind, boot_results)

  map(metrics, function(m) {
    vals <- boot_matrix[, m]
    lower <- as.numeric(quantile(vals, alpha / 2, na.rm = TRUE))
    upper <- as.numeric(quantile(vals, 1 - alpha / 2, na.rm = TRUE))

    if (m == "brier") {
      lower <- max(0, lower)
      upper <- min(1, upper)
    }

    list(lower = lower, upper = upper, se = sd(vals, na.rm = TRUE))
  }) %>%
    set_names(metrics)
}


# =============================================================================
# Helper: Summary Statistics
# =============================================================================

#' Compute summary statistics across all validation clusters
#' @keywords internal
compute_summary_stats <- function(results, metrics) {

  metric_labels <- c(
    auc = "AUC",
    brier = "Brier Score",
    cal_intercept = "Calibration Intercept",
    cal_slope = "Calibration Slope"
  )

  map_dfr(metrics, function(m) {
    vals <- results[[m]]
    tibble(
      metric = metric_labels[m],
      mean = mean(vals, na.rm = TRUE),
      sd = sd(vals, na.rm = TRUE),
      median = median(vals, na.rm = TRUE),
      min = min(vals, na.rm = TRUE),
      max = max(vals, na.rm = TRUE)
    )
  })
}


# =============================================================================
# S3 Methods: print, summary, plot
# =============================================================================

#' Print method for iecv_result
#' @export
print.iecv_result <- function(x, ...) {
  cat("Internal-External Cross-Validation Results\n")
  cat("==========================================\n\n")
  cat("Model:", get_model_label(x$model_type), "\n")
  cat("Clusters:", x$n_clusters, "\n")
  cat("Metrics:", paste(x$metrics, collapse = ", "), "\n")
  cat("Bootstrap replicates:", x$n_boot, "\n")
  cat("Confidence level:", x$conf_level * 100, "%\n\n")

  cat("Summary across clusters:\n")
  print(x$summary, n = Inf)

  invisible(x)
}


#' Summary method for iecv_result
#' @export
summary.iecv_result <- function(object, ...) {
  cat("IECV Summary\n")
  cat("============\n\n")
  cat("Model:", get_model_label(object$model_type), "\n\n")

  cat("Per-cluster results:\n\n")

  # Format table with CIs
  display_df <- object$cluster_results %>%
    select(id, n_validation, n_events)

  for (m in object$metrics) {
    ci_col <- sprintf(
      "%.3f (%.3f-%.3f)",
      object$cluster_results[[m]],
      object$cluster_results[[paste0(m, "_lower")]],
      object$cluster_results[[paste0(m, "_upper")]]
    )
    display_df[[m]] <- ci_col
  }

  print(display_df, n = Inf)

  cat("\n\nPooled summary:\n\n")
  print(object$summary, n = Inf)

  invisible(object)
}


#' Plot method for iecv_result
#'
#' @param x An iecv_result object
#' @param type Type of plot: "all" (default), "auc", "brier", "cal_intercept",
#'   "cal_slope", "calibration", "dca", or "shap" (tree models only)
#' @param smooth Logical; for DCA plots, whether to smooth the curves. Default is TRUE.
#' @param max_display For SHAP plots, maximum number of features to display. Default is 10.
#' @param ... Additional arguments passed to plotting functions
#'
#' @export
plot.iecv_result <- function(x, type = "all", smooth = TRUE, max_display = 10, ...) {
  is_tree_model <- x$model_type %in% c("xgboost", "lightgbm")
  valid_types <- c("all", x$metrics, "calibration", "dca")
  if (is_tree_model) valid_types <- c(valid_types, "shap")

  if (!type %in% valid_types) {
    stop(
      sprintf("Invalid plot type '%s'. Valid options: %s", type, paste(valid_types, collapse = ", ")),
      call. = FALSE
    )
  }

  switch(
    type,
    all = plot_all_metrics(x),
    calibration = plot_calibration_pooled(x, ...),
    dca = plot_dca_pooled(x, smooth = smooth, ...),
    shap = {
      if (!is_tree_model) {
        stop("SHAP plots are only available for tree-based models.", call. = FALSE)
      }
      plot_shap(x, max_display = max_display, ...)
    },
    plot_metric_forest(x, metric = type)
  )
}


# =============================================================================
# Plotting Functions
# =============================================================================

#' Forest plot for a single metric
#' @keywords internal
plot_metric_forest <- function(x, metric, cluster_order = NULL) {

  results <- x$cluster_results
  conf_pct <- x$conf_level * 100

  cfg <- list(
    auc = list(label = "AUC", ref = 0.5, ref_label = "No discrimination"),
    brier = list(label = "Brier Score", ref = 0.25, ref_label = "Reference"),
    cal_intercept = list(label = "Calibration Intercept", ref = 0, ref_label = "No miscalibration"),
    cal_slope = list(label = "Calibration Slope", ref = 1, ref_label = "Perfect calibration")
  )[[metric]]

  lower_col <- paste0(metric, "_lower")
  upper_col <- paste0(metric, "_upper")
  pooled <- mean(results[[metric]], na.rm = TRUE)

  if (is.null(cluster_order)) {
    cluster_order <- sort(unique(results$id))
  }
  results <- results %>%
    mutate(id = factor(id, levels = rev(cluster_order)))

  ggplot(results, aes(x = .data[[metric]], y = id)) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(xmin = .data[[lower_col]], xmax = .data[[upper_col]]),
      height = 0.2,
      orientation = "y"
    ) +
    geom_vline(xintercept = pooled, linetype = "dashed", color = "blue") +
    geom_vline(xintercept = cfg$ref, linetype = "dotted", color = "gray50") +
    labs(
      x = sprintf("%s (%d%% CI)", cfg$label, conf_pct),
      y = "Validation Cluster",
      title = sprintf("%s Across External Validations (%s)",
                      cfg$label, get_model_label(x$model_type)),
      subtitle = sprintf("Blue dashed = pooled (%.3f) | Gray dotted = %s", pooled, cfg$ref_label)
    ) +
    theme_minimal() +
    theme(
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 10)
    )
}


#' Combined forest plot for all metrics
#' @keywords internal
plot_all_metrics <- function(x) {

  cluster_order <- sort(unique(x$cluster_results$id))

  plots <- lapply(x$metrics, function(m) {
    plot_metric_forest(x, m, cluster_order = cluster_order) +
      ggtitle(NULL) +
      theme(plot.subtitle = element_blank())
  })

  n_plots <- length(plots)
  ncol <- if (n_plots <= 2) n_plots else 2

  patchwork::wrap_plots(plots, ncol = ncol)
}


#' Pooled calibration plot using probably package
#' @keywords internal
plot_calibration_pooled <- function(x,
                                    smooth = TRUE,
                                    include_rug = TRUE,
                                    include_ribbon = TRUE,
                                    conf_level = 0.9,
                                    ...) {

  cal_data <- x$predictions %>%
    mutate(
      .outcome = factor(outcome, levels = c(1, 0), labels = c("event", "non_event")),
      .pred_event = predicted_prob
    )

  probably::cal_plot_logistic(
    cal_data,
    truth = .outcome,
    estimate = .pred_event,
    smooth = smooth,
    include_rug = include_rug,
    include_ribbon = include_ribbon,
    conf_level = conf_level,
    event_level = "first",
    ...
  ) +
    labs(
      title = sprintf("Calibration Plot (%s)", get_model_label(x$model_type)),
      subtitle = sprintf("N = %d observations from %d clusters (pooled out-of-fold)",
                         nrow(cal_data), x$n_clusters)
    )
}


#' Prepare DCA data from predictions
#' @keywords internal
prepare_dca_data <- function(x) {
  x$predictions %>%
    transmute(outcome = outcome, predicted_prob = predicted_prob)
}


#' Decision Curve Analysis plot using dcurves package
#' @keywords internal
plot_dca_pooled <- function(x, smooth = TRUE, ...) {
  dca_data <- prepare_dca_data(x)
  dca_result <- dcurves::dca(
    outcome ~ predicted_prob,
    data = dca_data,
    label = list(predicted_prob = get_model_label(x$model_type)),
    ...
  )

  plot(dca_result, smooth = smooth) +
    labs(
      title = sprintf("Decision Curve Analysis (%s)", get_model_label(x$model_type)),
      subtitle = sprintf("N = %d observations from %d clusters (pooled out-of-fold)",
                         nrow(dca_data), x$n_clusters)
    )
}


# =============================================================================
# SHAP Functions for Tree Models
# =============================================================================

#' Create SHAP values for tree-based models
#'
#' Computes SHAP values using the shapviz package for XGBoost and LightGBM models.
#' Both models use their native TreeSHAP implementation for fast computation.
#'
#' @param x An iecv_result object from a tree-based model
#' @param n_samples Number of samples to use for SHAP calculation. Default is 500.
#'   Set to NULL to use all data (may be slow for large datasets).
#'
#' @return A shapviz object
#'
#' @details
#' Both XGBoost and LightGBM models use their native TreeSHAP implementations
#' via the shapviz package, providing fast and exact SHAP value computation.
#'
#' Note: Due to how tidymodels encodes factor outcomes, the raw tree model
#' predictions are for P("no"). SHAP values are automatically negated to
#' represent contributions to P("yes") (the event of interest).
#'
#' @export
get_shap <- function(x, n_samples = NULL) {
  assert_tree_model(x)

  if (!requireNamespace("shapviz", quietly = TRUE)) {
    stop("Package 'shapviz' is required. Install with: install.packages('shapviz')", call. = FALSE)
  }

  if (is.null(n_samples)) {
    n_samples <- 500
  }

  pred_data <- sample_predictor_data(x$predictor_data, n_samples)
  underlying_model <- workflows::extract_fit_parsnip(x$final_model)$fit

  if (x$model_type == "xgboost") {
    shp <- shapviz::shapviz(underlying_model, X_pred = data.matrix(pred_data), X = pred_data)
  } else if (x$model_type == "lightgbm") {
    shp <- shapviz::shapviz(underlying_model, X_pred = data.matrix(pred_data), X = pred_data)
  }


  # Negate SHAP values: tidymodels factor encoding causes raw model to predict

  # P("no"), but we want SHAP values for P("yes") (the event of interest).
  # Negating flips the direction so positive SHAP = higher event probability.
  shp$S <- -shp$S
  shp$baseline <- -shp$baseline

  shp
}


#' Sample predictor data for SHAP calculations
#' @keywords internal
sample_predictor_data <- function(pred_data, n_samples) {
  if (!is.null(n_samples) && nrow(pred_data) > n_samples) {
    set.seed(123)
    pred_data[sample(nrow(pred_data), n_samples), ]
  } else {
    pred_data
  }
}


#' Plot SHAP summary for tree-based models
#'
#' Creates a SHAP summary plot (beeswarm or bar) showing feature importance
#' and effects.
#'
#' @param x An iecv_result object from a tree-based model
#' @param plot_type Type of SHAP plot: "beeswarm" (default) or "bar"
#' @param max_display Maximum number of features to display. Default is 10.
#' @param n_samples Number of samples to use for SHAP calculation. Default is 500.
#' @param ... Additional arguments passed to shapviz plotting functions
#'
#' @return A ggplot object
#'
#' @keywords internal
plot_shap <- function(x, plot_type = "beeswarm", max_display = 10, n_samples = 500, ...) {
  if (!plot_type %in% c("beeswarm", "bar")) {
    stop("plot_type must be 'beeswarm' or 'bar'.", call. = FALSE)
  }

  shp <- get_shap(x, n_samples = n_samples)
  subtitle <- if (plot_type == "beeswarm") "Feature effects on model predictions" else "Mean |SHAP value|"

  shapviz::sv_importance(shp, kind = plot_type, max_display = max_display, ...) +
    labs(
      title = sprintf("SHAP Feature Importance (%s)", get_model_label(x$model_type)),
      subtitle = subtitle
    )
}


#' Plot SHAP dependence for a specific feature
#'
#' Creates a SHAP dependence plot showing the relationship between a feature
#' and its SHAP values.
#'
#' @param x An iecv_result object from a tree-based model
#' @param feature Name of the feature to plot
#' @param color_feature Name of feature for coloring points. Default is "auto"
#'   which selects the feature with highest interaction.
#' @param n_samples Number of samples to use for SHAP calculation. Default is 500.
#' @param ... Additional arguments passed to sv_dependence
#'
#' @return A ggplot object
#' @export
plot_shap_dependence <- function(x, feature, color_feature = "auto", n_samples = 500, ...) {
  assert_tree_model(x)

  shapviz::sv_dependence(get_shap(x, n_samples), v = feature, color_var = color_feature, ...) +
    labs(
      title = sprintf("SHAP Dependence: %s (%s)", feature, get_model_label(x$model_type)),
      subtitle = "Effect of feature value on prediction"
    )
}


# =============================================================================
# Variable Importance Function
# =============================================================================

#' Extract variable importance from IECV results
#'
#' For logistic regression, returns odds ratios with confidence intervals.
#' For tree-based models, returns SHAP-based importance or native importance.
#'
#' @param x An iecv_result object
#' @param type For tree models: "shap" (default) or "native".
#'   "shap" computes mean |SHAP values|, "native" uses model-specific importance.
#' @param n_samples For SHAP importance, number of samples to use. Default is 500.
#'
#' @return A tibble with variable importance measures
#' @export
variable_importance <- function(x, type = "shap", n_samples = 500) {
  assert_iecv_result(x)

  if (x$model_type == "logistic") {
    return(importance_logistic(x))
  }

  if (!type %in% c("shap", "native")) {
    stop("type must be 'shap' or 'native'.", call. = FALSE)
  }

  if (type == "native") {
    importance_native(x)
  } else {
    importance_shap(x, n_samples)
  }
}


#' Logistic regression importance (odds ratios)
#' @keywords internal
importance_logistic <- function(x) {
  broom::tidy(x$final_model, exponentiate = TRUE, conf.int = TRUE) %>%
    filter(term != "(Intercept)") %>%
    arrange(desc(abs(log(estimate)))) %>%
    select(
      variable = term,
      odds_ratio = estimate,
      p_value = p.value,
      conf_low = conf.low,
      conf_high = conf.high
    )
}


#' Native model importance
#' @keywords internal
importance_native <- function(x) {
  underlying_model <- workflows::extract_fit_parsnip(x$final_model)$fit

  if (x$model_type == "xgboost") {
    imp <- xgboost::xgb.importance(model = underlying_model)
    tibble(
      variable = imp$Feature,
      importance = imp$Gain,
      frequency = imp$Frequency,
      cover = imp$Cover
    ) %>%
      arrange(desc(importance))
  } else if (x$model_type == "lightgbm") {
    imp <- lightgbm::lgb.importance(model = underlying_model)
    tibble(
      variable = imp$Feature,
      importance = imp$Gain,
      frequency = imp$Frequency,
      cover = imp$Cover
    ) %>%
      arrange(desc(importance))
  }
}


#' SHAP-based importance
#' @keywords internal
importance_shap <- function(x, n_samples) {
  shp <- get_shap(x, n_samples = n_samples)
  imp <- shapviz::sv_importance(shp, kind = "bar", show_numbers = TRUE)

  imp$data %>%
    as_tibble() %>%
    arrange(desc(value)) %>%
    select(variable = feature, mean_abs_shap = value)
}


# =============================================================================
# Utility Functions
# =============================================================================

#' Format metric with confidence interval
#'
#' @param estimate Point estimate
#' @param lower Lower CI bound
#' @param upper Upper CI bound
#' @param digits Number of decimal places
#'
#' @return Character string formatted as "estimate (lower-upper)"
#' @export
format_ci <- function(estimate, lower, upper, digits = 3) {
  sprintf(
    "%.*f (%.*f-%.*f)",
    digits, estimate,
    digits, lower,
    digits, upper
  )
}


#' Extract final model coefficients from IECV results (logistic only)
#'
#' @param x An iecv_result object
#' @param exponentiate Logical; if TRUE, returns odds ratios. Default is TRUE.
#' @param conf.int Logical; if TRUE, includes confidence intervals. Default is TRUE.
#'
#' @return A tibble of model coefficients
#' @export
tidy_final_model <- function(x, exponentiate = TRUE, conf.int = TRUE) {
  assert_iecv_result(x)

  if (x$model_type != "logistic") {
    message("Note: tidy_final_model() is for logistic regression. For tree models, use variable_importance().")
    return(variable_importance(x, type = "native"))
  }

  broom::tidy(x$final_model, exponentiate = exponentiate, conf.int = conf.int)
}


#' Extract Decision Curve Analysis table from IECV results
#'
#' @param x An iecv_result object
#' @param thresholds Numeric vector of threshold probabilities.
#' @param ... Additional arguments passed to dcurves::dca()
#'
#' @return A tibble with DCA results
#' @export
dca_table <- function(x, thresholds = seq(0, 0.99, by = 0.01), ...) {
  assert_iecv_result(x)

  dca_result <- dcurves::dca(
    outcome ~ predicted_prob,
    data = prepare_dca_data(x),
    thresholds = thresholds,
    label = list(predicted_prob = get_model_label(x$model_type)),
    ...
  )

  dcurves::as_tibble(dca_result) %>%
    select(variable, label, threshold, n, net_benefit, tp_rate, fp_rate) %>%
    arrange(variable, threshold)
}


#' Get DCA object for further analysis
#'
#' @param x An iecv_result object
#' @param ... Additional arguments passed to dcurves::dca()
#'
#' @return A dca object from the dcurves package
#' @export
get_dca <- function(x, ...) {
  assert_iecv_result(x)

  dcurves::dca(
    outcome ~ predicted_prob,
    data = prepare_dca_data(x),
    label = list(predicted_prob = get_model_label(x$model_type)),
    ...
  )
}
