## Multivariate Methods

## TASK 1 -------
train_data <- read.csv("train_data.csv", header=TRUE)
sapply(train_data, class)

test_data <- read.csv("test_data.csv", header=TRUE)

future_data <- read.csv("future_data.csv", header=TRUE)

## Simple linear model --------------------------------------------------------

# Fit a linear model including all 
# explanatory variables of our training set to the training data.

fit.lm <- lm(y~., data=train_data)  
## Alternatively, fit.lm <- lm(y~(x1+x2+x3+x4+x5+x6+x7+x8), data=train_data)

print(fit.lm)

# Outputs from lm for our test set, exclude the labels. 
output.lm <- predict(fit.lm, test_data[, 1:8]) 
output.lm

# Predict class labels of our test data from lm. 
pred.lm <- ifelse(output.lm>0.5, 1, 0)

# Confusion table 
tab.lm <- table(test_data[,9], pred.lm)
print(tab.lm)

# Classification error 
error.lm <- 1- sum(diag(tab.lm))/ sum(rowSums(tab.lm))
cat("Classification error for linear model=", error.lm*100, "percent\n")


## K-nearest neighbors ------------------------------------------------------
library(class)

# Use the 9th column of the training set as the class argument in the function knn(). 
target.category <- train_data[, 9]

## What is the best value of k?

# Initialization 
k_vector <- rep(0, 200)
class.accuracy.vector <- rep(0, 200)
iter <- 0

library(class)

for(k in seq(1, 401, by=2)){
  knn.cv <- knn.cv(train_data[, 1:8], target.category, k)
  tab <- table(target.category, knn.cv)
  classification.accuracy <- 100*sum(diag(tab))/ sum(tab)
  
  iter <- iter+1
  k_vector[iter] <- k
  class.accuracy.vector[iter] <- classification.accuracy
}

# Plot the results 
plot(k_vector, class.accuracy.vector, ann=F)
title(xlab="k", ylab="Classification accuracy", main="LOOCV error")
abline(v=9, col="red", lty=2)

k_vector[which.max(class.accuracy.vector)]
max(class.accuracy.vector)
## We choose k to be 9 as this gives us the highest classification accuracy.

# Run the knn function.
pred.knn <- knn(train_data[,1:8], test_data[,1:8], cl=target.category, k=9)

# Confusion table

# Extract the labels of the test data to measure accuracy.  
test.category <- test_data[,9]

tab.knn <- table(pred.knn, test.category)

# Classification error 
error.knn <- 1- sum(diag(tab.knn))/ sum(tab.knn)
cat("Classification error for k-nearest neighbours=", error.knn*100, "percent\n")


## Canonical Variate Analysis (CVA) -----------------------------------------------------------

library(MASS)
# Fit cva to training data by setting the prior probabilities of the lda to 50/50. 
cva <- lda(train_data[, -9], train_data[, 9], prior=c(0.5,0.5))
print(cva)
plot(cva)

test.cva <- predict(cva, newdata=test_data[, -9])

test.cva$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.cva <- table(test_data[, 9], test.cva$class)
print(tab.cva)

# Classification error 
error.cva <- 1-sum(diag(tab.cva))/sum(tab.cva)
cat("Classification error for LDA=", error.cva*100,"percent\n")


## Alternatively, 
# Separate the training measurements into the two responses.
# Only keep the explanatory variables, but remove the labels. 

train.gp1 <- train_data[train_data$y=="0", -9]
train.gp2 <- train_data[train_data$y=="1", -9]


# Calculating the pooled covariance matrix. 
n1 <- nrow(train.gp1)
n2 <- nrow(train.gp2)

S.gp1 <- var(train.gp1)
S.gp2 <- var(train.gp2)

S.pooled <- ((n1-1)*S.gp1 + (n2-1)*S.gp2)/ (n1+n2-2)

# Calculating the canonical vector. 
mean.gp1 <- apply(train.gp1, 2, mean)
mean.gp2 <- apply(train.gp2, 2, mean)

mean.diff <- mean.gp1 - mean.gp2

canvector <- solve(S.pooled)%*%mean.diff

# Calculating the canonical variate scores on the test set. 
z <- t(canvector)%*%t(test_data[, -9])

# Build a classifier. 
z.gp.midpoint <- (mean.diff %*% solve(S.pooled)%*%(mean.gp1+mean.gp2))/2
z.gp.midpoint <- as.vector(z.gp.midpoint)

# Classify.  
z.class <- rep(NA, nrow(test_data))
z.class[z < z.gp.midpoint] <- "1"
z.class[z > z.gp.midpoint] <- "0"
print(z.class)

# Classification table 
tab.cva <- table(z.class, test_data[, 9])
print(tab.cva)

# Classification error 
error.cva <- 1 - sum(diag(tab.cva))/ sum(tab.cva)
cat("Classification error for CVA=", error.cva*100, "percent\n")


## Linear discriminant analysis -----------------------------------------------------------------
library(MASS)

# Fit lda to training data, allowing lda to estimate the prior probabilities from the data. 
lda <- lda(train_data[, -9], train_data[, 9])
print(lda)
plot(lda)

test.ld <- predict(lda, newdata=test_data[, -9])

test.ld$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.lda <- table(test_data[, 9], test.ld$class)
print(tab.lda)

# Classification error 
error.lda <- 1-sum(diag(tab.lda))/sum(tab.lda)
cat("Classification error for LDA=", error.lda*100,"percent\n")


## Quadratic discriminant analysis ---------------------------------------------------------------
library(MASS)

# Fit qda to training data, allowing qda to estimate the prior probabilities from the data. 
qda <- qda(train_data[, -9], train_data[, 9])
print(qda)

test.qd <- predict(qda, newdata=test_data[, -9])
names(test.qd)

test.qd$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.qda <- table(test_data[, 9], test.qd$class)
print(tab.qda)

# Classification error 
error.qda <- 1-sum(diag(tab.qda))/sum(tab.qda)
cat("Classification error for QDA=", error.qda*100,"percent\n")


## Un-pruned classification trees ---------------------------------------------------------------------
library(rpart)
library(rpart.plot)

# Fit the tree using training data. 
fit.tree <- rpart(y~., data=train_data, method="class")
summary(fit.tree)

# Predictions. 
tree.pred <- predict(fit.tree, newdata=test_data[, -9], type="class")

# Confusion table. 
tree.tab <- table(test_data[,9], tree.pred)
print(tree.tab)

# Classification error 
tree.error <- 1-sum(diag(tree.tab))/ sum(tree.tab)
cat("Classification error for un-pruned classification tree = ", 100*tree.error, "percent\n")


## Pruned classification tree -------------------------------------------------------------------
printcp(fit.tree)
plotcp(fit.tree)


fit.tree.prune <- rpart(y~., data=train_data, method="class", cp=0.091)

# Classification table 
tree.pred.prune <- predict(fit.tree.prune, newdata=test_data[, -9], type="class")
tree.tab.prune <- table(test_data[,9], tree.pred.prune)
print(tree.tab.prune)

# Classification error 
tree.error.prune <- 1-sum(diag(tree.tab.prune))/ sum(tree.tab.prune)
cat("Classification error for pruned classification tree= ", 100*tree.error.prune, "percent\n")


## Random forest ---------------------------------------------------------------------------------
library(randomForest)
library(tidyverse)
library(caret)

# As per note, convert the binary disease labels into factors. 
train_data$y <- as.factor(train_data$y)
test_data$y <- as.factor(test_data$y)

# Create Random forest from training set
set.seed(1)
model.RF <- train(y~., data=train_data, method="rf", 
                  trControl=trainControl("cv", number=20))

model.RF$finalModel$mtry # Optimal number of variables sampled at each split. Optimized using cross validation. 

# Predict class of test set
predict.RF <- predict(model.RF, newdata=test_data[, -9])

# Classification table 
RF.predict <- table(test_data[,9], predict.RF)
print(RF.predict)

# Classification error 
RF.error <- 1-sum(diag(RF.predict))/ sum(RF.predict)
cat("Classification error of random forest = ", 100*RF.error, "percent\n")

# Convert the binary disease factors back into labels after using random forest.  
train_data$y <- as.numeric(train_data$y)-1
test_data$y <- as.numeric(test_data$y)-1
## Note that we have to take away one when converting the labels back to numeric, otherwise we will get values 1 and 2 rather than 0 and 1. 

## TASK 2 -----------------------------------------------------------------------------------

## Standardise the training data ----

# The column means of the training data 
train.mean <- apply(train_data[, 1:8], 2, mean)
print(train.mean)

# The column standard deviations of the training data 
train.sd <- apply(train_data[, 1:8], 2, sd)
print(train.sd)

# Standardise the training data 
train_data_centred <- sapply(1:8, function(i)train_data[, i]-train.mean[i])
head(train_data_centred)
train_data_standard <- sapply(1:8, function(i)train_data_centred[, i]/train.sd[i])
head(train_data_standard)
head(train_data)

# Ensure that standardise training data is a data frame containing all relevant data.  
train_data_standard <- as.data.frame(train_data_standard)
train_data_standard <- transform(train_data_standard, y=train_data[,9])
colnames(train_data_standard) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y")

## Standardise the test data -----

# Note that the mean and standard deviations are NOT recomputed using the test data. 
# These values will be taken from the training data. 

test_data_centred <- sapply(1:8, function(i)test_data[,i]-train.mean[i])
head(test_data_centred)
test_data_standard <- sapply(1:8, function(i)test_data_centred[,i]/train.sd[i])

# Ensure that standardise test data is a data frame containing all relevant data.  
test_data_standard <- as.data.frame(test_data_standard)
test_data_standard <- transform(test_data_standard, y=test_data[,9])
colnames(test_data_standard) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "y")

train_data2 <-  train_data_standard
test_data2 <- test_data_standard


## Simple linear model --------------------------------------------------------

# Fit a linear model including all 
# explanatory variables of our training set to the training data.

fit.lm2 <- lm(y~., data=train_data2)  
## Alternatively, fit.lm <- lm(y~(x1+x2+x3+x4+x5+x6+x7+x8), data=train_data)

print(fit.lm2)

# Outputs from lm for our test set, exclude the labels. 
output.lm2 <- predict(fit.lm2, test_data2[, 1:8]) 
output.lm2

# Predict class labels of our test data from lm. 
pred.lm2 <- ifelse(output.lm2>0.5, 1, 0)

# Confusion table 
tab.lm2 <- table(test_data2[,9], pred.lm2)
print(tab.lm2)

# Classification error 
error.lm2 <- 1- sum(diag(tab.lm2))/ sum(rowSums(tab.lm2))
cat("Classification error for linear model=", error.lm2*100, "percent\n")


## K-nearest neighbors ------------------------------------------------------
library(class)

# Use the 9th column of the training set as the class argument in the function knn(). 
target.category2 <- train_data2[, 9]

## What is the best value of k?

# Initialization 
k_vector2 <- rep(0, 200)
class.accuracy.vector2 <- rep(0, 200)
iter2 <- 0

library(class)

for(k in seq(1, 401, by=2)){
  knn.cv2 <- knn.cv(train_data2[, 1:8], target.category2, k)
  tab2 <- table(target.category2, knn.cv2)
  classification.accuracy2 <- 100*sum(diag(tab2))/ sum(tab2)
  
  iter2 <- iter2+1
  k_vector2[iter2] <- k
  class.accuracy.vector2[iter2] <- classification.accuracy2
}

# Plot the results 
plot(k_vector2, class.accuracy.vector2, ann=F)
title(xlab="k", ylab="Classification accuracy", main="LOOCV error")
abline(v=47, col="red", lty=2)

k_vector2[which.max(class.accuracy.vector2)]
max(class.accuracy.vector2)
## We choose k to be 47 as this gives us the highest classification accuracy.

# Run the knn function.
pred.knn2 <- knn(train_data2[,1:8], test_data2[,1:8], cl=target.category2, k=47)

# Confusion table
test.category2 <- test_data2[, 9]
tab.knn2 <- table(pred.knn2, test.category2)

# Classification error 
error.knn2 <- 1- sum(diag(tab.knn2))/ sum(tab.knn2)
cat("Classification error for k-nearest neighbours=", error.knn2*100, "percent\n")


## Canonical Variate Analysis (CVA) -----------------------------------------------------------

library(MASS)
# Fit cva to training data by setting the prior probabilities of the lda to 50/50. 
cva2 <- lda(train_data2[, -9], train_data2[, 9], prior=c(0.5,0.5))
print(cva2)
plot(cva2)

test.cva2 <- predict(cva2, newdata=test_data2[, -9])

test.cva2$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.cva2 <- table(test_data2[, 9], test.cva2$class)
print(tab.cva2)

# Classification error 
error.cva2 <- 1-sum(diag(tab.cva2))/sum(tab.cva2)
cat("Classification error for LDA=", error.cva2*100,"percent\n")


## Linear discriminant analysis -----------------------------------------------------------------
library(MASS)

# Fit lda to training data, allowing lda to estimate the prior probabilities from the data. 
lda2 <- lda(train_data2[, -9], train_data2[, 9])
print(lda2)
plot(lda2)

test.ld2 <- predict(lda2, newdata=test_data2[, -9])

test.ld2$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.lda2 <- table(test_data2[, 9], test.ld2$class)
print(tab.lda2)

# Classification error 
error.lda2 <- 1-sum(diag(tab.lda2))/sum(tab.lda2)
cat("Classification error for LDA=", error.lda2*100,"percent\n")


## Quadratic discriminant analysis ---------------------------------------------------------------
library(MASS)

# Fit qda to training data, allowing qda to estimate the prior probabilities from the data. 
qda2 <- qda(train_data2[, -9], train_data2[, 9])
print(qda2)

test.qd2 <- predict(qda2, newdata=test_data2[, -9])
names(test.qd2)

test.qd2$posterior   # Posterior probability of being in each class for the first predictions. 

# Classification table 
tab.qda2 <- table(test_data2[, 9], test.qd2$class)
print(tab.qda2)

# Classification error 
error.qda2 <- 1-sum(diag(tab.qda2))/sum(tab.qda2)
cat("Classification error for QDA=", error.qda2*100,"percent\n")


## Un-pruned classification trees ---------------------------------------------------------------------
library(rpart)
library(rpart.plot)

# Fit the tree using training data. 
fit.tree2 <- rpart(y~., data=train_data2, method="class")
summary(fit.tree2)

# Predictions. 
tree.pred2 <- predict(fit.tree2, newdata=test_data2[, -9], type="class")

# Confusion table. 
tree.tab2 <- table(test_data2[,9], tree.pred2)
print(tree.tab2)

# Classification error 
tree.error2 <- 1-sum(diag(tree.tab2))/ sum(tree.tab2)
cat("Classification error for un-pruned classification tree = ", 100*tree.error2, "percent\n")


## Pruned classification tree -------------------------------------------------------------------
printcp(fit.tree2)
plotcp(fit.tree2)

fit.tree.prune2 <- rpart(y~., data=train_data2, method="class", cp=0.091)

# Classification table 
tree.pred.prune2 <- predict(fit.tree.prune2, newdata=test_data2[, -9], type="class")
tree.tab.prune2 <- table(test_data2[,9], tree.pred.prune2)
print(tree.tab.prune2)

# Classification error 
tree.error.prune2 <- 1-sum(diag(tree.tab.prune2))/ sum(tree.tab.prune2)
cat("Classification error for pruned classification tree= ", 100*tree.error.prune2, "percent\n")


## Random forest ---------------------------------------------------------------------------------
library(randomForest)
library(tidyverse)
library(caret)

# As per note, convert the binary disease labels into factors. 
train_data2$y <- as.factor(train_data2$y)
test_data2$y <- as.factor(test_data2$y)

# Create Random forest from training set
set.seed(1)
model.RF2 <- train(y~., data=train_data2, method="rf", 
                   trControl=trainControl("cv", number=20))

model.RF2$finalModel$mtry # Optimal number of variables sampled at each split. Optimized using cross validation. 

# Predict class of test set
predict.RF2 <- predict(model.RF2, newdata=test_data2[, -9])

# Classification table 
RF.predict2 <- table(test_data2[,9], predict.RF2)
print(RF.predict2)

# Classification error 
RF.error2 <- 1-sum(diag(RF.predict2))/ sum(RF.predict2)
cat("Classification error of random forest = ", 100*RF.error2, "percent\n")

# Convert the binary disease factors back into labels after using random forest.  
train_data2$y <- as.numeric(train_data2$y)-1
test_data2$y <- as.numeric(test_data2$y)-1
## Note that we have to take away one when converting the labels back to numeric, otherwise we will get values 1 and 2 rather than 0 and 1. 


## Task 3 ------
# Tree without data scaling and without pruning. 
rpart.plot(fit.tree, type=2, extra=4)

# Tree without data scaling and with pruning. 
rpart.plot(fit.tree.prune, type=2, extra=4)

# Tree with data scaling and without pruning. 
rpart.plot(fit.tree2, type=2, extra=4)

# Tree with data scaling and with pruning. 
rpart.plot(fit.tree.prune2, type=2, extra=4)

## Task 4 ----

## Predictions WITHOUT scaling ----

# Linear regression model ----
output.lm.t <- predict(fit.lm, future_data) 
pred.lm.t <- ifelse(output.lm.t>0.5, 1, 0)
print(pred.lm.t)

# K-nearest neighbors ----
pred.knn.t <- knn(train_data[, 1:8], future_data, cl=target.category, k=9)
pred.knn.t

# Canonical variance analysis ----
test.cva.t <- predict(cva, newdata=future_data)
print(test.cva.t$class)

# Linear discriminant analysis ----
test.ld.t <- predict(lda, newdata=future_data)
print(test.ld.t$class)

# Quadratic discriminant analysis ----
test.qd.t <- predict(qda, newdata=future_data)
print(test.qd.t$class)

# Un-pruned classification tree ----
tree.pred.t <- predict(fit.tree, newdata=future_data, type="class")
print(tree.pred.t)

# Pruned classification tree ----
tree.pred.prune.t <- predict(fit.tree.prune, newdata=future_data, type="class")
print(tree.pred.prune.t)

# Random forest ----
predict.RF <- predict(model.RF, newdata=future_data)
print(predict.RF)


## Predictions WTIH scaling ---

future_data_centred <- sapply(1:8, function(i)future_data[, i]-train.mean[i])
head(future_data_centred)
future_data_standard <- sapply(1:8, function(i)future_data_centred[, i]/train.sd[i])
head(future_data_standard)

future_data2 <- as.data.frame(future_data_standard)
colnames(future_data2) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")

# Linear regression model ----
output.lm.t2 <- predict(fit.lm2, future_data2) 
pred.lm.t2 <- ifelse(output.lm.t2>0.5, 1, 0)
print(pred.lm.t2)

# K-nearest neighbors ----
predicted.class2.t <- knn(train_data2[, 1:8], future_data2, cl=target.category2, k=47)
predicted.class2.t

# Canonical variance analysis ----
test.cva.t2 <- predict(cva2, newdata=future_data2)
print(test.cva.t2$class)

# Linear discriminant analysis ----
test.ld2.t <- predict(lda2, newdata=future_data2)
print(test.ld2.t$class)

# Quadratic discriminant analysis ----
test.qd2.t <- predict(qda2, newdata=future_data2)
print(test.qd2.t$class)

# Un-pruned classification tree ----
tree.pred2.t <- predict(fit.tree2, newdata=future_data2, type="class")
print(tree.pred2.t)

# Pruned classification tree ----
tree.pred.prune2.t <- predict(fit.tree.prune2, newdata=future_data2, type="class")
print(tree.pred.prune2.t)

# Random forest ----
predict.RF2.t <- predict(model.RF2, newdata=future_data2)
print(predict.RF2.t)









