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
mod <- lm(motor_UPDRS ~ ., data=df_train_scaled)

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
loglikelihood <- function(data, model) {
  # Get variable names and compute predicted values from hat matrix multiplication
  y_var <- as.character(as.list(as.list(model$call)$formula)[[2]]) # https://stackoverflow.com/questions/9694255/extract-formula-from-model-in-r
  x_var <- names(model$coefficients)[2:length(model$coefficients)]
  y_hat <- as.matrix(cbind(1, data[, x_var])) %*% as.matrix(model$coefficients)
  
  # Get sigma and n observations
  sigma <- sd(data[, y_var])
  n <- nrow(data)
  
  # Compute log likelihood
  log_likelihood <- -n * log(sqrt(2 * pi) * sigma) - 1 / (2 * sigma^2) * sum((data[, y_var] - y_hat)^2)
  
  return(log_likelihood)
}

loglikelihood(df_train_scaled, mod)

# b
ridge <- function(data, model, lambda) {
  penalty <- lambda * norm(as.matrix(model$coefficients), "f") # https://stackoverflow.com/questions/10933945/how-to-calculate-the-euclidean-norm-of-a-vector-in-r
  min_loglikelihood <- -loglikelihood(data, model)
  penalized_min_loglikelihood <- min_loglikelihood + penalty
  
  return(pred_test)
}

# c
ridge_opt <- function() {
  
}

# d
df <- function() {
  
}