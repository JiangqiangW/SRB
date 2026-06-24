
setwd("C:\\R\\15ML+SHAP ")
library(caret)    # machine learning framework
library(pROC)     # ROC curve analysis
library(ggplot2)  # visualization
library(shapviz)  # SHAP analysis
library(kernelshap)
library(dplyr)    # data processing
library(rmda)
library(catboost)
library(lightgbm)
library(DALEX)
# library(remotes)
# library("devtools")

# 1. data loading
train_data <- read.csv('train_SRB.csv', header = T, row.names = 1)
test_data <- read.csv('test_SRB.csv', header = T, row.names = 1)

# 2. specify response variable name
response_var <- "Group"

# Set the control group to"Control" and the SRB group to "DN"

train_data$Group <- gsub("(.+)\\_(.+)\\_(.+)", '\\3', rownames(train_data))
table(train_data$Group)
test_group <- gsub("(.+)\\_(.+)\\_(.+)", '\\3', rownames(test_data))
test_group  <- ifelse(test_group  == 'Control', '0','1')
table(test_group)
set.seed(12345)


# 2. specify response variable name.

response_var <- "Group"

# 3. set cross-validation parameters.

cv_control <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# === performance metric calculation function ==========================================
calculate_metrics <- function(truth, probs, threshold = 0.5) {
  # convert probabilities to predicted labels
  pred <- factor(ifelse(probs >= threshold, "Positive", "Negative"), 
                 levels = c("Negative", "Positive"))
  
  # convert true labels to factors
  truth_fct <- factor(ifelse(truth == 1, "Positive", "Negative"), 
                      levels = c("Negative", "Positive"))
  
  # compute the confusion matrix
  cm <- confusionMatrix(pred, truth_fct, positive = "Positive")
  
  # extract all metrics
  metrics <- c(
     cm$byClass["Sensitivity"],  # sensitivity = TP/(TP+FN)
     cm$byClass["Specificity"],  # specificity = TN/(TN+FP)
    cm$overall["Accuracy"],       # accuracy = (TP+TN)/(TP+FP+TN+FN)
    cm$byClass["Pos Pred Value"],     # positive predictive value = TP/(TP+FP)
    cm$byClass["Neg Pred Value"],      # negative predictive value = TN/(TN+FN)
    cm$byClass["F1"]     
  )
   metrics["Youden"] <- metrics["Sensitivity"] + metrics["Specificity"] - 1  # Youden index
  return(metrics)
}
# ===============================================================



# validate data format
if(!response_var %in% colnames(train_data)) {
  stop(paste("响应变量", response_var, "不存在于训练数据中"))
}

# ensure the response variable is a factor
train_data[[response_var]] <- factor(train_data[[response_var]], 
                                     levels = c("Control", "DN"))
formula <- as.formula(paste(response_var, "~ ."))


model_settings <- data.frame(
  AlgorithmName = c("RandomForest", "GradientBoosting", "SVM_Kernel",
                     "BoostingMethod",
                     "BayesMethod", "XGBoost"),
  Implementation = c("rf", "xgbTree",  "svmRadial", 
                      "gbm",
                     "nb", "xgbLinear")
)

# create lists to store results

# store model performance information:
train_metrics <- list()
test_metrics <- list()

# store model ROC information:
train_roc_list <- list()  
test_roc_list <- list() 

modelContainer <- list()
AUCresults <- c()
training_times <- numeric()

# create a list to store predicted probabilities for DCA
train_probs_list <- list()
test_probs_list <- list()

# add residual calculation lists to the model training loop
residuals_train_list <- list()
residuals_test_list <- list()

best_tune_params <- list() # record grid hyperparameters
# start the overall timer
total_start <- Sys.time()

cat("===== Start model training ", format(total_start, "%Y-%m-%d %H:%M:%S"), " =====\n")

# model training and evaluation
for (idx in seq_len(nrow(model_settings))) {
  algoName <- model_settings$AlgorithmName[idx]
  algoImpl <- model_settings$Implementation[idx]
  
  # start the model timer
  start_time <- Sys.time()
  cat(sprintf("\n[%d/%d] 开始训练 %s: %s\n", idx, nrow(model_settings), algoName, format(start_time, "%H:%M:%S")))
  
  tryCatch({
    # notes for special models
    slow_models <- c("SVM_Kernel", "NeuralNet", "GradientBoosting", "AdaptiveBoosting", "XGBoost")
    if (algoName %in% slow_models) {
      cat("  ...\n")
    }
    
    # model training
    cat("  - Start model training...\n")
    
    # apply algorithm-specific settings
    if (algoName == "SVM_Kernel") {
      # SVM tuning
      svm_tuneGrid <- expand.grid(
        sigma = c(0.01, 0.1, 1),
        C = c(0.25, 0.5, 1, 2)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   prob.model = TRUE, trControl = cv_control,
                                   tuneGrid = svm_tuneGrid,metric = "ROC")

    } else if (algoName == "GradientBoosting") {
      
      xgbtree_tuneGrid <- expand.grid(nrounds = c(100, 200, 300),  max_depth = c(2, 4, 6),
        eta = c(0.03, 0.10),  gamma = c(0, 0.1),  colsample_bytree = 0.8,   min_child_weight = c(1, 5),
        subsample = 0.8
      )
      
      trainedModel <- caret::train(formula,data = train_data,method = "xgbTree",trControl = cv_control,
        tuneGrid = xgbtree_tuneGrid,  metric = "ROC",   verbose = FALSE
      )
    } else if (algoName == "RandomForest") {
      # random forest tuning
      rf_tuneGrid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = rf_tuneGrid,metric = "ROC")
    } else if (algoName == "BoostingMethod") {
      # GBM tuning parameters
      gbm_tuneGrid <- expand.grid(
        n.trees = c(50, 100, 150),
        interaction.depth = c(1, 3, 5),
        shrinkage = c(0.01, 0.1),
        n.minobsinnode = c(5, 10)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = gbm_tuneGrid,metric = "ROC")
    } else if (algoName == "BayesMethod") {
      # naive Bayes tuning parameters
      nb_tuneGrid <- expand.grid(
        fL = c(0, 0.5, 1),
        usekernel = c(TRUE, FALSE),
        adjust = c(0.5, 1, 1.5)
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = nb_tuneGrid,metric = "ROC")
    } else if (algoName == "XGBoost") {
      # XGBoost linear model tuning parameters
      xgb_tuneGrid <- expand.grid(
        nrounds = c(50, 100, 150),  # boosting iterations
        lambda = c(0, 0.1, 1),      # L2 regularization
        alpha = c(0, 0.1, 1),       # L1 regularization
        eta = c(0.01, 0.1, 0.3)     # learning rate
      )
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,
                                   tuneGrid = xgb_tuneGrid,metric = "ROC")
    } else {
      # for models without tuning parameters(such as glm and lda, default training is sufficient)
      trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                   trControl = cv_control,metric = "ROC")
    }
    
    # calculate training time
    model_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  - Training complete (耗时: %s秒)\n", model_time))
    
    # +++ extract and print the best parameters +++
    cat("  - 最佳参数组合:\n")
    print(trainedModel$bestTune)
    
    # store the best parameters
    best_tune_params[[algoName]] <- trainedModel$bestTune
    
    # add evaluation information
    cat("  - 开始模型评估...\n")
    
    # === training set performance metrics =========================================
    # get cross-validation predicted probabilities
    cv_probs <- trainedModel$pred[trainedModel$pred$rowIndex, "DN"]
    cv_indices <- trainedModel$pred[trainedModel$pred$rowIndex, "rowIndex"]
    
    # extract true labels
    truth_train <- ifelse(train_data[[response_var]][cv_indices] == "DN", 1, 0)
    
    # compute training set metrics
    train_metrics[[algoName]] <- calculate_metrics(truth_train, cv_probs)
    
    # create the training set ROC curve object
    roc_train <- roc(truth_train, cv_probs)
    train_roc_list[[algoName]] <- roc_train
    
    # === test set performance metrics =========================================
    test_probs <- predict(trainedModel, newdata = test_data, type = "prob")[, "DN"]
    
    # compute test set metrics
    test_metrics[[algoName]] <- calculate_metrics(test_group, test_probs)
    
    # create the test set ROC curve object
    roc_test <- roc(test_group, test_probs)
    test_roc_list[[algoName]] <- roc_test
    
    # store model/AUC results
    modelContainer[[algoImpl]] <- trainedModel
    AUCresults <- c(AUCresults, paste0(algoName, ": ", 
                                       sprintf("%.03f", roc_test$auc)))  # save AUC
    
    # store predicted probabilities for DCA
    # training set: create a complete probability vector
    train_probs_full <- rep(NA, nrow(train_data))
    train_probs_full[trainedModel$pred$rowIndex] <- trainedModel$pred$DN
    train_probs_list[[algoName]] <- train_probs_full
    
    # test set: store directly
    test_probs_list[[algoName]] <- test_probs
    
    # performance output
    cat(sprintf("  - 训练集AUC: %.3f\n", auc(roc_train)))
    cat(sprintf("  - 测试集AUC: %.3f\n", auc(roc_test)))
    
    # record training time
    training_times[algoName] <- model_time
    
  }, error = function(e) {
    # record the error message and elapsed time
    error_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    cat(sprintf("  ! 错误: %s (耗时: %s秒)\n", e$message, error_time))
    
    # record the error message
    train_metrics[[algoName]] <- rep(NA, 7)
    test_metrics[[algoName]] <- rep(NA, 7)
    training_times[algoName] <- error_time
    # store NA to indicate parameter extraction failed
    best_tune_params[[algoName]] <- NA
  })
  
  # display the completion status for each model
  cat(sprintf("[%d/%d] 完成 %s | 累计耗时: %.1f秒\n", idx, nrow(model_settings), algoName, 
              as.numeric(difftime(Sys.time(), total_start, units = "secs"))))
  
  # add a separator line
  cat("-----------------------------------------\n")
}
# final report
total_time <- round(as.numeric(difftime(Sys.time(), total_start, units = "mins")), 1)
cat("\n===== 训练完成! 总耗时: ", total_time, "分钟 =====\n")



# two special models need to be trained separately.
# === train the CatBoost model separately after the model training loop ========================
cat("\n=====  CatBoost  =====\n")

# 1. prepare the data format required by CatBoost
train_features <- train_data[, -which(names(train_data) == response_var)]
train_labels <- ifelse(train_data[[response_var]] == "DN", 1, 0)

test_features <- test_data
test_labels <- test_group  # already preprocessed as 0/1 above

# create the CatBoost Pool object
train_features <- as.matrix(train_features)
mode(train_features) <- "numeric"
train_pool <- catboost.load_pool(data = train_features, label = as.numeric(train_labels))
test_features <- as.matrix(test_features)
mode(test_features) <- "numeric"
test_labels <- as.numeric(test_labels)
test_pool <- catboost.load_pool(data = test_features, label = test_labels)


# 2. define the parameter grid
catboost_grid <- expand.grid(
  depth = c(4, 6, 8),          # tree depth
  learning_rate = c(0.01, 0.05, 0.1),  # learning rate
  l2_leaf_reg = c(1, 3, 5)     # L2 regularization coefficient
)

# initialize the best parameters and performance
best_auc <- 0
best_params <- NULL
best_iter <- NULL

# 3. grid search
cat("-- 开始网格搜索寻找最佳参数 --\n")
for (i in 1:nrow(catboost_grid)) {
  params <- as.list(catboost_grid[i, ])
  
  # set base parameters
  cv_params <- list(
    iterations = 500,
    learning_rate = params$learning_rate,
    depth = params$depth,
    l2_leaf_reg = params$l2_leaf_reg,
    loss_function = 'Logloss',
    eval_metric = 'AUC',
    early_stopping_rounds = 50,
    random_seed = 42
  )
  
  # run cross-validation
  cv_results <- catboost.cv(
    pool = train_pool,
    params = cv_params,
    fold_count = 5,
    partition_random_seed = 42,
    shuffle = TRUE
  )
  
  # get the best iteration count and AUC
  iter <- which.max(cv_results$test.AUC.mean)
  auc_value <- max(cv_results$test.AUC.mean)
  
  cat(sprintf("参数组合 %d/%d: depth=%d, lr=%.3f, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
              i, nrow(catboost_grid), params$depth, params$learning_rate, 
              params$l2_leaf_reg, iter, auc_value))
  
  # update the best parameters
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- cv_params
    best_iter <- iter
  }
}

# 4. train the final model using the best parameters
cat(sprintf("\n最佳参数: depth=%d, lr=%.3f, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
            best_params$depth, best_params$learning_rate, 
            best_params$l2_leaf_reg, best_iter, best_auc))
# add CatBoost best parameters to the parameter summary list
catboost_best_params <- data.frame(
  depth = best_params$depth,
  learning_rate = best_params$learning_rate,
  l2_leaf_reg = best_params$l2_leaf_reg,
  iterations = best_iter
)
best_tune_params[["CATBoost"]] <- catboost_best_params

# set final model parameters
catboost_params <- best_params
catboost_params$iterations <- best_iter

# 5. train the CatBoost model and time it
start_time_catboost <- Sys.time()
cat(sprintf("开始训练 CatBoost: %s\n", format(start_time_catboost, "%H:%M:%S")))

catboost_model <- catboost.train(
  learn_pool = train_pool,
  params = catboost_params
)

# calculate training time
catboost_train_time <- round(as.numeric(difftime(Sys.time(), start_time_catboost, units = "secs")), 1)
cat(sprintf("CatBoost 训练完成 (耗时: %s秒)\n", catboost_train_time))

# 4. get predicted probabilities
# training set predicted probabilities
train_catboost_preds <- catboost.predict(catboost_model, train_pool, prediction_type = "Probability")
# test set predicted probabilities
test_catboost_preds <- catboost.predict(catboost_model, test_pool, prediction_type = "Probability")

# 5. compute performance metrics and integrate them into the result lists
algoName <- "CATBoost"

# training set metrics
train_truth <- ifelse(train_data[[response_var]] == "DN", 1, 0)
train_metrics[[algoName]] <- calculate_metrics(train_truth, train_catboost_preds)

# test set metrics
test_metrics[[algoName]] <- calculate_metrics(as.numeric(test_labels), test_catboost_preds)

# create the ROC object
roc_train_catboost <- roc(train_truth, train_catboost_preds)
roc_test_catboost <- roc(as.numeric(test_labels), test_catboost_preds)

# add to the ROC list
train_roc_list[[algoName]] <- roc_train_catboost
test_roc_list[[algoName]] <- roc_test_catboost

# add to the model container and AUC results
modelContainer[["catboost"]] <- catboost_model
AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_catboost$auc)))

# add to the probability list(for DCA)
train_probs_list[[algoName]] <- train_catboost_preds
test_probs_list[[algoName]] <- test_catboost_preds

# === add CatBoost residual calculation ===
# training set residuals = observed values - predicted probabilities
train_residuals_catboost <- train_truth - train_catboost_preds
# test set residuals = observed values - predicted probabilities
test_residuals_catboost <- as.numeric(test_labels) - test_catboost_preds

# store results
residuals_train_list[[algoName]] <- train_residuals_catboost
residuals_test_list[[algoName]] <- test_residuals_catboost

# record training time
training_times[algoName] <- catboost_train_time

# performance output
cat(sprintf("  - CatBoost 训练集AUC: %.3f\n", auc(roc_train_catboost)))
cat(sprintf("  - CatBoost 测试集AUC: %.3f\n", auc(roc_test_catboost)))


# === train the LightGBM model separately after the model training loop ======================
cat("\n===== 开始单独训练 LightGBM 模型 =====\n")
# 1. prepare the data format required by LightGBM
train_features_lgb <- train_data[, -which(names(train_data) == response_var)]
train_labels_lgb <- ifelse(train_data[[response_var]] == "DN", 1, 0)

test_features_lgb <- test_data
test_labels_lgb <- as.numeric(test_group)

# process categorical variables
categorical_cols <- names(train_features_lgb)[sapply(train_features_lgb, is.factor)]

# convert to matrix format
train_matrix_lgb <- as.matrix(train_features_lgb)
test_matrix_lgb <- as.matrix(test_features_lgb)

# 2. create the LightGBM dataset
dtrain_lgb <- lgb.Dataset(
  data = train_matrix_lgb,
  label = train_labels_lgb,
  categorical_feature = categorical_cols
)

dtest_lgb <- lgb.Dataset(
  data = test_matrix_lgb,
  label = test_labels_lgb,
  reference = dtrain_lgb,
  categorical_feature = categorical_cols
)

# 3. define the parameter grid
lgb_grid <- expand.grid(
  num_leaves = c(15, 31, 63),          # number of leaf nodes
  learning_rate = c(0.01, 0.05, 0.1),  # learning rate
  min_data_in_leaf = c(5, 10, 20),     # minimum data per leaf 
  lambda_l2 = c(0, 0.1, 0.5),          # L2 regularization coefficient
  feature_pre_filter= F
)

# initialize the best parameters and performance
best_auc <- 0
best_params <- NULL
best_iter <- NULL

# 4. grid search
cat("-- 开始网格搜索寻找最佳参数 --\n")
for (i in 1:nrow(lgb_grid)) {
  params <- as.list(lgb_grid[i, ])
  
  lgb_params <- list(
    objective = "binary",
    metric = "auc",
    num_leaves = params$num_leaves,
    learning_rate = params$learning_rate,
    min_data_in_leaf = params$min_data_in_leaf,
    lambda_l2 = params$lambda_l2,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 5,
    verbosity = -1,
    seed = 42,
    feature_pre_filter = FALSE  # key setting here
  )
  
  # cross-validation
  cv_model <- lgb.cv(
    params = lgb_params,
    data = dtrain_lgb,
    nrounds = 500,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 50,
    eval_freq = 10
  )
  
  # get the best iteration count and AUC
  iter <- cv_model$best_iter
  auc_value <- max(unlist(cv_model$record_evals$valid$auc$eval))
  
  cat(sprintf("参数组合 %d/%d: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
              i, nrow(lgb_grid), params$num_leaves, params$learning_rate, 
              params$min_data_in_leaf, params$lambda_l2, iter, auc_value))
  
  # update the best parameters
  if (auc_value > best_auc) {
    best_auc <- auc_value
    best_params <- lgb_params
    best_iter <- iter
  }
}

# 5. train the final model using the best parameters
cat(sprintf("\n最佳参数: leaves=%d, lr=%.3f, min_data=%d, l2=%.1f | 最佳迭代: %d, AUC: %.4f\n",
            best_params$num_leaves, best_params$learning_rate, 
            best_params$min_data_in_leaf, best_params$lambda_l2, best_iter, best_auc))

# add LightGBM best parameters to the parameter summary list
lightgbm_best_params <- data.frame(
  num_leaves = best_params$num_leaves,
  learning_rate = best_params$learning_rate,
  min_data_in_leaf = best_params$min_data_in_leaf,
  lambda_l2 = best_params$lambda_l2,
  nrounds = best_iter
)
best_tune_params[["LightGBM"]] <- lightgbm_best_params

# 6. train the LightGBM model and time it
start_time_lgb <- Sys.time()
cat(sprintf("开始训练 LightGBM: %s\n", format(start_time_lgb, "%H:%M:%S")))

lgb_model <- lgb.train(
  params = best_params,
  data = dtrain_lgb,
  nrounds = best_iter
)

# calculate training time
lgb_train_time <- round(as.numeric(difftime(Sys.time(), start_time_lgb, units = "secs")), 1)
cat(sprintf("LightGBM 训练完成 (耗时: %s秒)\n", lgb_train_time))

# 7. get predicted probabilities
# training set predicted probabilities
train_lgb_preds <- predict(lgb_model, train_matrix_lgb)
# test set predicted probabilities
test_lgb_preds <- predict(lgb_model, test_matrix_lgb)

# 8. compute performance metrics and integrate them into the result lists
algoName <- "LightGBM"

# training set metrics
train_metrics[[algoName]] <- calculate_metrics(train_labels_lgb, train_lgb_preds)

# test set metrics
test_metrics[[algoName]] <- calculate_metrics(test_labels_lgb, test_lgb_preds)

# create the ROC object
roc_train_lgb <- roc(train_labels_lgb, train_lgb_preds)
roc_test_lgb <- roc(test_labels_lgb, test_lgb_preds)

# add to the ROC list
train_roc_list[[algoName]] <- roc_train_lgb
test_roc_list[[algoName]] <- roc_test_lgb

# add to the model container and AUC results
modelContainer[["lightgbm"]] <- lgb_model
AUCresults <- c(AUCresults, paste0(algoName, ": ", sprintf("%.03f", roc_test_lgb$auc)))

# add to the probability list(for DCA)
train_probs_list[[algoName]] <- train_lgb_preds
test_probs_list[[algoName]] <- test_lgb_preds

# === add LightGBM residual calculation ===
# training set residuals = observed values - predicted probabilities
train_residuals_lgb <- train_labels_lgb - train_lgb_preds
# test set residuals = observed values - predicted probabilities
test_residuals_lgb <- test_labels_lgb - test_lgb_preds

# store results
residuals_train_list[[algoName]] <- train_residuals_lgb
residuals_test_list[[algoName]] <- test_residuals_lgb

# record training time
training_times[algoName] <- lgb_train_time

# performance output
cat(sprintf("  - LightGBM 训练集AUC: %.3f\n", auc(roc_train_lgb)))
cat(sprintf("  - LightGBM 测试集AUC: %.3f\n", auc(roc_test_lgb)))



# display the training time table
cat("\n模型训练时间汇总:\n")
print(data.frame(
  Model = names(training_times),
  Train_time_seconds = unlist(training_times)
))



# if you previously removed any models, also remove the corresponding colors here.
# prepare the mapping between models and colors
color_mapping <- c(
  BayesMethod = "#20B2AA",
  BoostingMethod = "#FFA500",
  CATBoost = 'green',
  GradientBoosting = "#98FB98",
  LightGBM = 'purple',
  RandomForest = "#FF00FF",
  SVM_Kernel = "#FA8072"
)

allcolour <- data.frame(color_mapping)$color_mapping

# === performance metric tables ==============================================
library(gridExtra)
library(grid)

# organize training set metrics into a data frame
train_results <- bind_rows(train_metrics, .id = "Model")

# rename the training set metrics table
colnames(train_results) <- c("Models", 
                             "Sensitivity", "Specificity", "Accuracy", 
                             "PPV", "NPV", 'F1', "Youden's index")

# save training set performance metrics
write.csv(train_results, "2a_Train_Performance_Metrics0620.csv", row.names = FALSE)

# create a PDF table of training set performance metrics
pdf("2a_Train_Performance_Table.pdf", width = 12, height = 10)
grid.table(train_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()

# === plot the training set metric line chart ===
library(ggplot2)
library(tidyr)

# convert data to long format
train_long <- gather(train_results, key = "Metric", value = "Value", -Models)

# create the training set metric line chart
train_plot <- ggplot(train_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Training Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        legend.position = "bottom",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = color_mapping)
train_plot 
# save the training set metric line chart
ggsave("3a_Train_Metrics_LinePlot.pdf", train_plot, width = 10, height = 8)
dev.off()
# organize test set metrics into a data frame
test_results <- bind_rows(test_metrics, .id = "Model")

# rename the test set metrics table
colnames(test_results) <- c("Models", 
                            "Sensitivity", "Specificity", "Accuracy", 
                            "PPV", "NPV", 'F1', "Youden's index")

# save test set performance metrics
write.csv(test_results, "2b_Test_Performance_Metrics0620.csv", row.names = FALSE)

# create a PDF table of test set performance metrics
pdf("2b_Test_Performance_Table.pdf", width = 12, height = 10)
grid.table(test_results, 
           rows = NULL,
           theme = ttheme_default(
             core = list(bg_params = list(fill = c("#F7F7F7", "#FFFFFF"), col = "gray"),
                         fg_params = list(cex = 0.8)),
             colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
           ))
dev.off()



test_long <- gather(test_results, key = "Metric", value = "Value", -Models)

# create the test set metric line chart
test_plot <- ggplot(test_long, aes(x = Metric, y = Value, group = Models, color = Models)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Validation Set Performance Metrics", 
       x = "Performance Metrics", 
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 12),
        legend.position = "bottom",
        legend.title = element_text(size = 12, face = "bold"),
        legend.text = element_text(size = 11),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")) +
  scale_y_continuous(limits = c(-1, 1)) +
  scale_color_manual(values = color_mapping)
test_plot 
# save the test set metric line chart
ggsave("3b_Test_Metrics_LinePlot.pdf", test_plot, width = 10, height = 8)
dev.off()

# === AUC confidence intervals ==============================================
# compute training set AUC confidence intervals
train_auc_ci <- data.frame()
for (model_name in names(train_roc_list)) {
  if (!is.null(train_roc_list[[model_name]])) {
    ci <- ci.auc(train_roc_list[[model_name]])
    train_auc_ci <- rbind(train_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(train_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# format training set AUC and confidence intervals
train_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                               train_auc_ci$AUC,
                               train_auc_ci$CI_lower,
                               train_auc_ci$CI_upper)

# save training set AUC confidence interval results
write.csv(train_auc_ci, "4a_Train_AUC_Confidence_Intervals0620.csv", row.names = FALSE)

# compute test set AUC confidence intervals
test_auc_ci <- data.frame()
for (model_name in names(test_roc_list)) {
  if (!is.null(test_roc_list[[model_name]])) {
    ci <- ci.auc(test_roc_list[[model_name]])
    test_auc_ci <- rbind(test_auc_ci, data.frame(
      Model = model_name,
      AUC = auc(test_roc_list[[model_name]]),
      CI_lower = ci[1],
      CI_upper = ci[3]
    ))
  }
}

# format test set AUC and confidence intervals
test_auc_ci$AUC_CI <- sprintf("%.3f (%.3f-%.3f)", 
                              test_auc_ci$AUC,
                              test_auc_ci$CI_lower,
                              test_auc_ci$CI_upper)

# save test set AUC confidence interval results
write.csv(test_auc_ci, "4b_Test_AUC_Confidence_Intervals0620.csv", row.names = FALSE)


# === forest plot: AUC comparison ======================================
# prepare the data required for the forest plot
train_forest_df <- train_auc_ci %>%
  arrange(AUC)%>% # sort by AUC in ascending order
  mutate(
    Model = factor(Model, levels = rev(Model)),  # reverse the model order so the first item is at the top
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 


# create the forest plot
ggplot(train_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +  # diamond points represent AUC values
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +  # reference line
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3,check_overlap = TRUE, show.legend = FALSE) +  # AUC value labels
  
  # styling settings
  scale_x_continuous(limits = c(min(train_forest_df$CI_lower) * 0.95, max(train_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  
  # titles and labels
  labs(title = "Forest Plot of Each Model AUC Score in Trainset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  
  # theme settings
  theme_minimal(base_size = 14) 
# save the forest plot
ggsave("5a_Trainset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")

test_forest_df <- test_auc_ci %>%
  arrange(AUC)%>% # sort by AUC in ascending order
  mutate(
    Model = factor(Model, levels = rev(Model)),  # reverse the model order so the first item is at the top
    AUC_label = sprintf("%.3f (%.3f-%.3f)", AUC, CI_lower, CI_upper)
  ) 



# create the forest plot
ggplot(test_forest_df, aes(x = AUC, y = Model)) +
  geom_point(aes(color = Model), size = 5, shape = 18) +  # diamond points represent AUC values
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper, color = Model),
                 height = 0.1, size = 1.0) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray", size = 0.8) +  # reference line
  geom_text(aes(label = AUC_label), 
            size = 4, hjust = 0.5, nudge_y = 0.3,check_overlap = TRUE, show.legend = FALSE) +  # AUC value labels
  
  # styling settings
  scale_x_continuous(limits = c(min(test_forest_df$CI_lower) * 0.95, max(test_forest_df$CI_upper) * 1.05),
                     breaks = seq(0.5, 1.0, by = 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  scale_color_manual(values = color_mapping) +
  
  # titles and labels
  labs(title = "Forest Plot of Each Model AUC Score in Testset",
       x = "AUC Score (95% CI)",
       y = "Models") +
  
  # theme settings
  theme_minimal(base_size = 14) 
# save the forest plot
ggsave("5b_Testset_Forest_Plot_AUC.pdf", width = 12, height = 8, device = "pdf")
dev.off()


# === decision curve analysis(DCA)module =====================================
# prepare training set data(convert to 0/1 format)
train_truth <- ifelse(train_data$Group == "DN", 1, 0)

# create the DCA data frame for the training set
dca_train_data <- data.frame(truth = train_truth)
for (model_name in names(train_probs_list)) {
  dca_train_data[[model_name]] <- train_probs_list[[model_name]]
}

# create the DCA data frame for the validation set
dca_test_data <- data.frame(truth = test_group)
for (model_name in names(test_probs_list)) {
  dca_test_data[[model_name]] <- test_probs_list[[model_name]]
}

# === custom DCA calculation function ===
calculate_dca <- function(data, outcome, predictors) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  results <- data.frame()
  
  # extract key statistics
  outcome_vector <- data[[outcome]]
  n <- length(outcome_vector)
  n_positive <- sum(outcome_vector == 1)  # number of actual positive cases
  n_negative <- sum(outcome_vector == 0)  # number of actual negative cases
  
  for (pt in thresholds) {
    # correctly compute the Treat all strategy
    all_positive_nb <- (n_positive / n) - (n_negative / n) * (pt / (1 - pt))
    
    # Treat none strategy (always 0)
    all_negative_nb <- 0
    
    # NB calculation for each model
    model_nbs <- sapply(predictors, function(p) {
      pred <- data[[p]]
      if (all(is.na(pred))) return(NA)
      
      # calculate model TP and FP
      pred_positive <- pred >= pt  # model predicts positive
      tp <- sum(outcome_vector == 1 & pred_positive, na.rm = TRUE)
      fp <- sum(outcome_vector == 0 & pred_positive, na.rm = TRUE)
      n_valid <- length(which(!is.na(pred)))
      
      (tp / n_valid) - (fp / n_valid) * (pt / (1 - pt))
    })
    
    # merge results
    res_row <- data.frame(
      threshold = pt,
      variable = c("Treat all", "Treat none", names(model_nbs)),
      net_benefit = c(all_positive_nb, all_negative_nb, model_nbs)
    )
    
    results <- rbind(results, res_row)
  }
  
  return(results)
}

# === run DCA using the custom function ===

# process training set data
dca_train_data_clean <- na.omit(dca_train_data)
model_names_train <- setdiff(colnames(dca_train_data_clean), "truth")
dca_train_res <- calculate_dca(dca_train_data_clean, "truth", model_names_train)

# process test set data
dca_test_data_clean <- na.omit(dca_test_data)
model_names_test <- setdiff(colnames(dca_test_data_clean), "truth")
dca_test_res <- calculate_dca(dca_test_data_clean, "truth", model_names_test)

# reshape DCA data to wide format
convert_dca_to_wide <- function(dca_data) {
  # extract unique strategy names
  strategies <- unique(dca_data$variable)
  
  # initialize the data frame
  wide_data <- data.frame(threshold = unique(dca_data$threshold))
  
  # create columns for each strategy
  for (strategy in strategies) {
    wide_data[[strategy]] <- sapply(wide_data$threshold, function(t) {
      dca_data$net_benefit[dca_data$threshold == t & dca_data$variable == strategy]
    })
  }
  
  return(wide_data)
}

# convert training and validation data
dca_train_wide <- convert_dca_to_wide(dca_train_res)
colnames(dca_train_wide) <- gsub(" ", "_", colnames(dca_train_wide))
dca_test_wide <- convert_dca_to_wide(dca_test_res)
colnames(dca_test_wide) <- gsub(" ", "_", colnames(dca_test_wide))

# Modified DCA plotting function with English-only labels
plot_custom_dca <- function(dca_wide, title) {
  # Extract model names (exclude all and none)
  model_names <- setdiff(colnames(dca_wide), c("threshold", "Treat_all", "Treat_none"))
  
  # Create base plot
  p <- ggplot(dca_wide, aes(x = threshold)) +
    # Plot reference strategies
    geom_line(aes(y = Treat_all, color = "Treat_all"), linetype = "dashed", size = 0.8) +
    geom_line(aes(y = Treat_none, color = "Treat_none"), linetype = "dashed", size = 0.8)
  
  # Add lines for each model
  for (model_name in model_names) {
    p <- p + geom_line(aes_string(y = model_name, color = shQuote(model_name)), size = 1)
  }
  
  # Number of models for color palette
  n_colors <- length(model_names)
  
  # Set colors and labels
  p <- p +
    scale_color_manual(
      name = "Strategy",
      values = c(
        "Treat_all" = "gray",
        "Treat_none" = "black",
        setNames(rainbow(n_colors), model_names)
      ),
      breaks = c("Treat_all", "Treat_none", model_names)
    ) +
    labs(
      title = title,
      x = "High risk threshold",
      y = "Net benefit"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.text = element_text(size = 9),
      legend.title = element_text(size = 10, face = "bold")
    ) +
    theme(panel.grid.major = element_line(color = "gray90", size = 0.2),
          panel.grid.minor = element_line(color = "gray95", size = 0.1)) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1), 
                       limits = c(0, 1)) +
    scale_y_continuous(limits = c(-0.1, max(dca_wide[model_names], na.rm = TRUE) * 1.1))
  
  return(p)
}

pdf("7a_DCA_Training_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_train_wide, "Decision Curve Analysis (Training Set)"))+scale_color_manual(values = color_mapping)
dev.off()

pdf("7b_DCA_Validation_Set.pdf", width=10, height=7)
print(plot_custom_dca(dca_test_wide, "Decision Curve Analysis (Validation Set)"))+scale_color_manual(values = color_mapping)
dev.off()



AUCresults
# === find the best model ===
best_model_name <- test_auc_ci$Model[which.max(test_auc_ci$AUC)]
# best_model_name <- "CATBoost"
cat(sprintf("最佳模型: %s (AUC = %.3f)\n", best_model_name, max(test_auc_ci$AUC)))

# convert to lowercase first, then use mutually exclusive checks
best_model_name <- tolower(best_model_name)
# get the best model object and handle the prediction function
if (best_model_name == "catboost") {
  final_model <- modelContainer[["catboost"]]
  
  # CatBoost-specific prediction function
  predict_catboost <- function(model, newdata) {
    pool <- catboost.load_pool(newdata)
    probs <- catboost.predict(model, pool, prediction_type = "Probability")
    pred_labels <- ifelse(probs > 0.5, "DN", "Control")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("Control", "DN"))))
  }
  
} else if (best_model_name == "lightgbm") {
  final_model <- modelContainer[["lightgbm"]]
  
  # LightGBM-specific prediction function
  predict_lightgbm <- function(model, newdata) {
    probs <- predict(model, as.matrix(newdata))
    pred_labels <- ifelse(probs > 0.5, "DN", "Control")
    return(list(probs = probs, labels = factor(pred_labels, levels = c("Control", "DN"))))
  }
  
} else {
  
  idx <- match(
    best_model_name,
    tolower(model_settings$AlgorithmName)
  )
  
  if (is.na(idx)) {
    stop("无法匹配模型名称：", best_model_name)
  }
  
  impl <- model_settings$Implementation[idx]
  final_model <- modelContainer[[impl]]
  
  if (is.null(final_model)) {
    stop("modelContainer中不存在模型：", impl)
  }
  
  predict_caret <- function(model, newdata) {
    probs <- predict(
      model,
      newdata = newdata,
      type = "prob"
    )[, "DN"]
    
    pred_labels <- predict(
      model,
      newdata = newdata
    )
    list(
      probs = probs,
      labels = pred_labels
    )
  }
}

predict_catboost <- function(model, newdata) {
  newdata <- as.matrix(newdata)
  mode(newdata) <- "numeric"  # force conversion to numeric
  pool <- catboost.load_pool(newdata)
  probs <- catboost.predict(model, pool, prediction_type = "Probability")
  pred_labels <- ifelse(probs > 0.5, "DN", "Control")
  return(list(probs = probs, labels = factor(pred_labels, levels = c("Control", "DN"))))
}



# SHAP visualization
#create a custom prediction function(output numeric probabilities)
# create a unified prediction function
# === SHAP visualization section ===
# create a unified prediction function
pred_wrapper <- function(model, model_type, newdata) {
  if (model_type == "caret") {
    return(predict(model, newdata = newdata, type = "prob")[, "DN"])
    
  } else if (model_type == "catboost") {
    newdata <- as.matrix(newdata)
    mode(newdata) <- "numeric"          # do not forget to force conversion to numeric
    pool <- catboost.load_pool(newdata)
    return(catboost.predict(model, pool, prediction_type = "Probability"))
    
  } else if (model_type == "lightgbm") {
    return(predict(model, as.matrix(newdata)))
    
  } else {
    stop("Unknown model_type: ", model_type)
  }
}


# # identify the best model type
best_model_type <- case_when(
  best_model_name %in% model_settings$AlgorithmName ~ "caret",
   best_model_name == "catboost" ~ "catboost",
   best_model_name == "LightGBM" ~ "lightgbm"
 )


 
print(class(final_model))
 
 ## ===== SHAP calculation =====
if (inherits(final_model, "train")) {                # caret framework
   shap_values <- kernelshap(
     final_model,
     X = train_data[, -ncol(train_data)],
     pred_fun = function(m, x) predict(m, as.data.frame(x), type = "prob")[, "DN"]
   )
   shap_vis <- shapviz(shap_values, train_data[, -ncol(train_data)])
   
 } else if (inherits(final_model, "catboost.Model")) { # CatBoost
   # the feature matrix must be numeric
   x_train <- as.matrix(train_data[, -which(names(train_data) == response_var)])
   mode(x_train) <- "numeric"
   pool_train <- catboost.load_pool(x_train)
   shap_values_matrix <- catboost.get_feature_importance(
     final_model, pool_train, type = "ShapValues"
   )
   shap_values <- shap_values_matrix[, -ncol(shap_values_matrix)]
   baseline <- shap_values_matrix[1, ncol(shap_values_matrix)]
   colnames(shap_values) <- colnames(x_train)
   shap_vis <- shapviz(shap_values, X = x_train, baseline = baseline)
   cat("CatBoost SHAP 计算完成\n")
   
 } else if (inherits(final_model, "lgb.Booster")) {   # LightGBM
   if (!requireNamespace("fastshap", quietly = TRUE)) install.packages("fastshap")
   library(fastshap)
   pred_fun <- function(object, newdata) predict(object, as.matrix(newdata))
   shap_values <- fastshap::explain(
     final_model,
     X = as.data.frame(train_data[, -which(names(train_data) == response_var)]),
     pred_wrapper = pred_fun,
     nsim = 10
   )
   shap_vis <- shapviz(shap_values,
                       X = train_data[, -which(names(train_data) == response_var)])
   cat("LightGBM SHAP 计算完成\n")
   
 } else {
   stop("final_model 类型无法识别：", class(final_model))
 }


feature_importance <- colMeans(abs(shap_vis$S))
 

sorted_features <- names(sort(feature_importance, decreasing = TRUE))

# visualization settings
visualization_theme <- theme_minimal() + 
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12))

print(best_model_name)      # should be "LogisticModel"
print(class(final_model))   # check the SHAP branch that just ran successfully
sv_importance
# Generate model interpretation plots - SHAP visualizations (English)

# 1. feature importance bar chart(global explanation)
#    show the average absolute impact of each feature on model output
#    larger values indicate a greater influence on model prediction
#    show exact values to facilitate quantitative comparison
pdf("13_SHAP_Feature_Importance_Barplot.pdf", width=8, height=6)
sv_importance(shap_vis, kind="bar", show_numbers=TRUE,max_display = 30L) +
  visualization_theme +
  labs(title = "Feature Importance (Mean Absolute SHAP)",
       subtitle = "Average impact magnitude of each feature on model predictions",
       x = "Mean |SHAP value|", y = "Feature",
       caption = "Bar height indicates feature importance, with value showing mean absolute SHAP")
dev.off()

# 2. beeswarm plot(feature effect distribution)
#    show the distribution of SHAP values for each feature across the entire dataset
#    color indicates feature value magnitude(red is high, blue is low)
#    show the relationship trend between feature values and predictions
#    more dispersed points indicate stronger nonlinearity
pdf("14_SHAP_BeeSwarm_Plot.pdf", width=8, height=8)
sv_importance(shap_vis, kind="bee", show_numbers=TRUE,max_display = 30L)+
  # scale_color_gradientn(colours = c("#3f4a8b", "white", "#d43a51"))  +
  visualization_theme +
  labs(title = "SHAP Value Distribution (Bee Swarm)",
       subtitle = "Each point represents one sample, color indicates feature value",
       x = "SHAP value (impact on model output)",
       y = "Feature",
       caption = "Red: high feature values | Blue: low feature values | Horizontal spread: direction of effect")
dev.off()

# 3. feature dependence plots(multiple features)
#    show the nonlinear relationship between feature values and their SHAP values
#    each subplot shows the dependency between one feature and model output
#    the trend line helps clarify the relationship pattern between the feature and the prediction
# version without lines
pdf("15_SHAP_Feature_1115.pdf", width=15, height=8)
sv_dependence(color_var = "auto",color = "#3b528b",shap_vis, sorted_features[11:15]) +  # Top 6 important features
  # scale_color_gradientn(
  #   colours = c("#3f4a8b", "white", "#d43a51")   # dark blue-white-red, commonly used in SSCI journals
  #   # or c("grey80", "grey20")                # monochrome gray
  # ) +
  # +
  geom_smooth(
    method = "loess",
    se = TRUE,           # whether to show the confidence interval
    color = "black",
    linetype = "dashed",
    size = 0.8
  ) +
  visualization_theme
dev.off()

pdf("15_SHAP_Feature_9.pdf", width = 6, height = 5)

sv_dependence(
  color_var = "auto",
  color     = "#3b528b",
  shap_vis,
  sorted_features[10]
) +
  scale_color_gradientn(colours = c("#3f4a8b", "white", "#d43a51")) +
  geom_smooth(
    method = "loess",
    span = 0.5,
    se     = TRUE,
    color  = "#FFA07A",
    linetype = "solid",
    size   = 0.8
  ) +
  visualization_theme

dev.off()


# 5. single-sample force plot
dir.create('17_Force_Plot', showWarnings = FALSE)
for (i in 1:min(10, nrow(train_data))) {
  pdf(paste0("17_Force_Plot/sample_", i, "_force.pdf"), width=9, height=6)
  p <- sv_force(shap_vis, row_id = i) +  
    labs(title = paste("Feature Contributions for Sample", i),
         subtitle = "Visual forces pushing prediction from base value to output")
  print(p)
  dev.off()
}



