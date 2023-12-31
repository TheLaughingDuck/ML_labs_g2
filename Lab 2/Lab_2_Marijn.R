library(tree)
library(ggplot2)

# 1.1
# Read in data and remove 'duration' column
df_bank <- read.csv("bank-full.csv", sep=";")
df_bank <- df_bank[, names(df_bank) != "duration"]

# Set categorical column as factor for modeling
# https://stackoverflow.com/questions/54768433/nas-introduced-by-coercion-error-when-using-tree-function
# https://www.listendata.com/2015/05/converting-multiple-numeric-variables.html
df_bank[sapply(df_bank, is.character)] <- lapply(df_bank[sapply(df_bank, is.character)], as.factor)

# Train/validation/test split
n <- nrow(df_bank)
set.seed(12345)
id_train <- sample(1:n, floor(n * 0.4))
df_train <- df_bank[id_train,]

id_rest <- setdiff(1:n, id_train)
set.seed(12345)
id_valid <- sample(id_rest, floor(n * 0.3))
df_valid <- df_bank[id_valid,]

id_test <- setdiff(id_rest, id_valid)
df_test <- df_bank[id_test,]

# 1.2a
n <- nrow(df_train)
mod1 <- tree(y ~ ., data=df_train)
pred_train1 <- predict(mod1, newdata=df_train, type="class")
pred_valid1 <- predict(mod1, newdata=df_valid, type="class")
misclass_train1 <- (table(df_train$y, pred_train1)["no", "yes"] + table(df_train$y, pred_train1)["yes", "no"]) / nrow(df_train)
misclass_valid1 <- (table(df_valid$y, pred_valid1)["no", "yes"] + table(df_valid$y, pred_valid1)["yes", "no"]) / nrow(df_valid)

plot(mod1)
text(mod1, pretty=0)

# 1.2b
# https://search.r-project.org/CRAN/refmans/tree/html/tree.control.html
mod2 <- tree(y ~ ., data=df_train, control=tree.control(nobs=n, minsize=7000))
pred_train2 <- predict(mod2, newdata=df_train, type="class")
pred_valid2 <- predict(mod2, newdata=df_valid, type="class")
misclass_train2 <- (table(df_train$y, pred_train2)["no", "yes"] + table(df_train$y, pred_train2)["yes", "no"]) / nrow(df_train)
misclass_valid2 <- (table(df_valid$y, pred_valid2)["no", "yes"] + table(df_valid$y, pred_valid2)["yes", "no"]) / nrow(df_valid)

plot(mod2)
text(mod2, pretty=0)

# 1.2c
# https://search.r-project.org/CRAN/refmans/tree/html/tree.control.html
mod3 <- tree(y ~ ., data=df_train, control=tree.control(nobs=n, mindev=0.0005))
pred_train3 <- predict(mod3, newdata=df_train, type="class")
pred_valid3 <- predict(mod3, newdata=df_valid, type="class")
misclass_train3 <- (table(df_train$y, pred_train3)["no", "yes"] + table(df_train$y, pred_train3)["yes", "no"]) / nrow(df_train)
misclass_valid3 <- (table(df_valid$y, pred_valid3)["no", "yes"] + table(df_valid$y, pred_valid3)["yes", "no"]) / nrow(df_valid)

plot(mod3)
text(mod3, pretty=0)

cat("Misclassification rate model a:\n", 
    "\tTrain: ", misclass_train1, "\n",
    "\tValidation: ", misclass_valid1, "\n\n",
    "Misclassification rate model b:\n",
    "\tTrain: ", misclass_train2, "\n",
    "\tValidation: ", misclass_valid2, "\n\n",
    "Misclassification rate model c:\n",
    "\tTrain: ", misclass_train3, "\n",
    "\tValidation: ", misclass_valid3,
    sep="")

# 2.3
train_score <- rep(0, 50)
valid_score <- rep(0, 50)

for (i in 2:50) {
  pruned_tree <- prune.tree(mod3, best=i)
  pred <- predict(pruned_tree, newdata=df_valid, type="tree")
  train_score[i] <- deviance(pruned_tree)
  valid_score[i] <- deviance(pred)
}

# https://r-coder.com/add-legend-r/
plot(2:50, train_score[2:50], type="b", col="red", ylim=c(8000, 12000), xlab="# leaves", ylab="Deviance")
points(2:50, valid_score[2:50], type="b", col="blue")
legend(x="topright", legend=c("Train", "Validation"), lty=c(1, 1), col=c("red", "blue"))

# 2.4
pruned_mod <- prune.tree(mod3, best=19)
pred_test <- predict(pruned_mod, newdata=df_test, type="class")

# Confusion matrix
# https://datascience.stackexchange.com/questions/27132/decision-tree-used-for-calculating-precision-accuracy-and-recall-class-breakd
conf_mat <- table(df_test$y, pred_test, dnn=c("True", "Pred"))
acc <- (conf_mat["yes", "yes"] + conf_mat["no", "no"]) / sum(conf_mat)
prec <- conf_mat["yes", "yes"] / (conf_mat["yes", "yes"] + conf_mat["no", "yes"])
rec <- conf_mat["yes", "yes"] / (conf_mat["yes", "yes"] + conf_mat["yes", "no"])
F1 <- (2 * prec * rec) / (prec + rec)

# 2.5
loss_matrix <- matrix(c(0, 1, 5, 0), nrow=2, ncol=2)
mod4 <- tree(y ~ ., data=df_train)
pruned_mod4 <- prune.tree(mod4, loss=loss_matrix)
plot(pruned_mod4)

# k=6 optimal
opt_pruned_mod4 <- prune.tree(mod4, k=6, loss=loss_matrix)
pred_train4 <- predict(opt_pruned_mod4, newdata=df_train, type="class")

table(df_train$y, pred_train4)
plot(opt_pruned_mod4)
text(opt_pruned_mod4, pretty=0)

# 2.6
v_pi <- seq(from=0.05, to=0.95, by=0.05)
log_mod <- glm(y ~ ., family=binomial(link="logit"), data=df_train)
pred_log_prob <- predict(log_mod, newdata=df_train, type="response")
pred_tree_prob <- predict(mod2, newdata=df_train, type="prob")

v_tpr_log <- vector()
v_fpr_log <- vector()
for (pi in v_pi) {
  pred_log_v <- ifelse(pred_log_prob > pi, "yes", "no")
  conf_mat_log <- table(df_train$y, pred_log_v)
  
  if (!("yes" %in% colnames(conf_mat_log))) {
    tpr_log <- 0
    fpr_log <- 0
  }
  else {
    tpr_log <- conf_mat_log["yes", "yes"] / sum(conf_mat_log)
    fpr_log <- conf_mat_log["no", "yes"] / sum(conf_mat_log)
  }
  
  v_tpr_log <- append(v_tpr_log, tpr_log)
  v_fpr_log <- append(v_fpr_log, fpr_log)
}

plot(v_fpr_log, v_tpr_log, type="l")
lines(c(0, 1), c(0, 1))
