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

# 1.2b
# https://search.r-project.org/CRAN/refmans/tree/html/tree.control.html
mod2 <- tree(y ~ ., data=df_train, control=tree.control(nobs=n, minsize=7000))
pred_train2 <- predict(mod2, newdata=df_train, type="class")
pred_valid2 <- predict(mod2, newdata=df_valid, type="class")
misclass_train2 <- (table(df_train$y, pred_train2)["no", "yes"] + table(df_train$y, pred_train2)["yes", "no"]) / nrow(df_train)
misclass_valid2 <- (table(df_valid$y, pred_valid2)["no", "yes"] + table(df_valid$y, pred_valid2)["yes", "no"]) / nrow(df_valid)

# 1.2c
# https://search.r-project.org/CRAN/refmans/tree/html/tree.control.html
mod3 <- tree(y ~ ., data=df_train, control=tree.control(nobs=n, mindev=0.0005))
pred_train3 <- predict(mod3, newdata=df_train, type="class")
pred_valid3 <- predict(mod3, newdata=df_valid, type="class")
misclass_train3 <- (table(df_train$y, pred_train3)["no", "yes"] + table(df_train$y, pred_train3)["yes", "no"]) / nrow(df_train)
misclass_valid3 <- (table(df_valid$y, pred_valid3)["no", "yes"] + table(df_valid$y, pred_valid3)["yes", "no"]) / nrow(df_valid)

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

# 2.6

