library(caret)

# Read in data
df_park <- read.csv("parkinsons.csv")

# 2.1
# Split training/test and scale
train_ind <- sample(1:nrow(df_park), floor(nrow(df_park) * 0.6))
df_train <- df_park[train_ind,]
df_test <- df_park[-train_ind,]

scaler <- preProcess(df_train)
df_train_scaled <- predict(scaler, df_train)
df_test_scaled <- predict(scaler, df_test)

# 2.2
# Train model
mod <- lm(motor_UPDRS ~ . + 0, data=df_train_scaled) # https://stats.stackexchange.com/questions/143155/doing-multiple-regression-without-intercept-in-r-without-changing-data-dimensio

# Predict values
pred_train <- data.frame(pred=predict(mod, df_train_scaled), act=df_train_scaled$motor_UPDRS)
pred_test <- data.frame(pred=predict(mod, df_test_scaled), act=df_test_scaled$motor_UPDRS)

# Compute MSE
# https://www.statology.org/how-to-calculate-mse-in-r/
mse_train <- mean((pred_train$act - pred_train$pred)^2)
mse_test <- mean((pred_test$act - pred_test$pred)^2)

print(paste0("MSE training data: ", mse_train, "\nMSE test data: ", mse_test))

# 2.3
# a
loglikelihood <- function(data, model, theta, sigma) {
  # Get variable names and y and x matrices
  y_var <- as.character(as.list(as.list(model$call)$formula)[[2]]) # https://stackoverflow.com/questions/9694255/extract-formula-from-model-in-r
  x_var <- names(model$coefficients)[2:length(model$coefficients)]
  
  y <- data[, y_var]
  x <- data[, x_var]
  
  # Get n observations
  n <- nrow(data)
  
  # Compute log likelihood
  log_likelihood <- -n * log(sqrt(2 * pi) * sigma) - 1 / (2 * sigma^2) * sum((y - t(theta) * x)^2)
  
  return(log_likelihood)
}

# b
ridge <- function(data, model, v_theta_sigma, lambda) {
  theta <- v_theta_sigma[1:length(v_theta_sigma) - 1]
  sigma <- v_theta_sigma[length(v_theta_sigma)]
  
  penalty <- lambda * norm(as.matrix(theta), "f") # https://stackoverflow.com/questions/10933945/how-to-calculate-the-euclidean-norm-of-a-vector-in-r
  min_loglikelihood <- -loglikelihood(data, model, theta, sigma)
  penalized_min_loglikelihood <- min_loglikelihood + penalty
  
  return(penalized_min_loglikelihood)
}

# c
ridge_opt <- function(data, model, lambda) {
  # # https://stackoverflow.com/questions/24623488/how-do-i-use-a-function-with-parameters-in-optim-in-r
  # https://stackoverflow.com/questions/59517244/r-optim-can-i-pass-a-list-to-parameter-par
  opt <- optim(c(rep(0, length(model$coefficients)), 1), ridge, data=data, model=model, lambda=lambda, method="BFGS")
  opt_theta <- opt$param[1:length(opt$param) - 1]
  opt_sigma <- opt$param[length(opt$param)]
}

# d
df <- function(data, model, lambda) {
  x_var <- names(model$coefficients)[2:length(model$coefficients)]
  x <- data[, x_var]
  
  # ERROR HERE
  ncol(x %*% solve(t(x) %*% x + lambda * diag(max(dim(x)))) %*% t(x)) # https://online.stat.psu.edu/stat508/lesson/5/5.1
}
