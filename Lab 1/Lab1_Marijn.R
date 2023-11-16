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
loglikelihood <- function(data, formula, theta, sigma) {
  # Get variable names and y and x matrices
  y_var <- all.vars(formula)[1]
  x_var <- all.vars(formula)[2:length(all.vars(formula))]
  
  if (x_var == ".") {
    x_var <- colnames(data)[colnames(data) != y_var]
  }
  
  y <- data[, y_var]
  x <- data[, x_var]
  
  # Get n observations
  n <- nrow(data)
  
  # Compute log likelihood
  log_likelihood <- -n * log(sqrt(2 * pi) * sigma) - 1 / (2 * sigma^2) * sum((y - t(theta) * x)^2)
  
  return(log_likelihood)
}

# b
ridge <- function(data, formula, v_theta_sigma, lambda) {
  theta <- v_theta_sigma[1:length(v_theta_sigma) - 1]
  sigma <- v_theta_sigma[length(v_theta_sigma)]
  
  penalty <- lambda * sum(theta^2) # https://stackoverflow.com/questions/10933945/how-to-calculate-the-euclidean-norm-of-a-vector-in-r
  min_loglikelihood <- -loglikelihood(data, formula, theta, sigma)
  penalized_min_loglikelihood <- min_loglikelihood + penalty
  
  return(penalized_min_loglikelihood)
}

# c
ridge_opt <- function(data, formula, lambda) {
  # # https://stackoverflow.com/questions/24623488/how-do-i-use-a-function-with-parameters-in-optim-in-r
  # https://stackoverflow.com/questions/59517244/r-optim-can-i-pass-a-list-to-parameter-par
  x_var <- all.vars(formula)[2:length(all.vars(formula))]
  
  if (x_var == ".") {
    x_var <- colnames(data)[colnames(data) != y_var]
  }
  
  opt <- optim(c(rep(0, length(x_var)), 1), ridge, data=data, formula=formula, lambda=lambda, method="BFGS")
  opt_theta <- opt$par[1:length(opt$par) - 1]
  opt_sigma <- opt$par[length(opt$par)]
  
  return(list(theta=opt_theta, sigma=opt_sigma))
}

# d
df <- function(data, formula, lambda) {
  # Get x matrix
  x_var <- all.vars(formula)[2:length(all.vars(formula))]
  
  if (x_var == ".") {
    x_var <- colnames(data)[colnames(data) != y_var]
  }
  
  x <- as.matrix(data[, x_var])
  
  # Compute trace of hat matrix
  sum(diag((x %*% solve(t(x) %*% x + lambda * diag(ncol(x))) %*% t(x)))) # https://online.stat.psu.edu/stat508/lesson/5/5.1
}

# 2.4
# Get opt theta and sigma for different lambdas
l_param1 <- ridge_opt(df_train_scaled, motor_UPDRS ~ ., lambda=1)
l_param100 <- ridge_opt(df_train_scaled, motor_UPDRS ~ ., lambda=100)
l_param1000 <- ridge_opt(df_train_scaled, motor_UPDRS ~ ., lambda=1000)

# Function for computing MSE
mse <- function(data, formula, theta) {
  # Get variable names and y and x matrices
  y_var <- all.vars(formula)[1]
  x_var <- all.vars(formula)[2:length(all.vars(formula))]
  
  if (x_var == ".") {
    x_var <- colnames(data)[colnames(data) != y_var]
  }
  
  y <- data[, y_var]
  x <- as.matrix(data[, x_var])
  
  y_hat <- x %*% theta
  
  # Compute MSE
  mean((y_hat - y)^2)
}

# Get y hat on training and test for different lambdas
mse_tr_1 <- mse(df_train_scaled, motor_UPDRS ~ ., l_param1$theta)
mse_te_1 <- mse(df_test_scaled, motor_UPDRS ~ ., l_param1$theta)

mse_tr_100 <- mse(df_train_scaled, motor_UPDRS ~ ., l_param100$theta)
mse_te_100 <- mse(df_test_scaled, motor_UPDRS ~ ., l_param100$theta)

mse_tr_1000 <- mse(df_train_scaled, motor_UPDRS ~ ., l_param1000$theta)
mse_te_1000 <- mse(df_test_scaled, motor_UPDRS ~ ., l_param1000$theta)

# Compute df for different models
df_tr_1 <- df(df_train_scaled, motor_UPDRS ~ ., 1)
df_te_1 <- df(df_test_scaled, motor_UPDRS ~ ., 1)

df_tr_100 <- df(df_train_scaled, motor_UPDRS ~ ., 100)
df_te_100 <- df(df_test_scaled, motor_UPDRS ~ ., 100)

df_tr_1000 <- df(df_train_scaled, motor_UPDRS ~ ., 1000)
df_te_1000 <- df(df_test_scaled, motor_UPDRS ~ ., 1000)

# Report output
cat("lambda = 1",
    "\nMSE train: ", mse_tr_1,
    "\nMSE test: ", mse_te_1,
    "\ndf train: ", df_tr_1,
    "\ndf test: ", df_te_1,
    "\n\n",
    "lambda = 100",
    "\nMSE train: ", mse_tr_100,
    "\nMSE test: ", mse_te_100,
    "\ndf train: ", df_tr_100,
    "\ndf test: ", df_te_100,
    "\n\n",
    "lambda = 1000",
    "\nMSE train: ", mse_tr_1000,
    "\nMSE test: ", mse_te_1000,
    "\ndf train: ", df_tr_1000,
    "\ndf test: ", df_te_1000,
    sep=""
)
