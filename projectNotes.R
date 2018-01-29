# Week 4 ProjectNotes
#  Practical Machine Learning


setwd("H:/Programming/Coursera/Machine Learning/project/")

require(caret)
require(randomForest)
#require(ElemStatLearn)
#require(pgmm)
require(rpart)
require(gbm)
#require(lubridate)
#require(forecast)
#require(e1071)
require(dplyr)
require(tictoc)

tic("loading data")
test_datafile <- "./data/pml-testing.csv"
train_datafile <- "./data/pml-training.csv"
train_fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
dir.create("data", showWarnings=FALSE)
if(!file.exists(train_datafile)) {
    download.file(train_fileURL, destfile=train_datafile)
}
if(!file.exists(test_datafile)) {
    download.file(test_fileURL, destfile=test_datafile)
}
raw_train <- read.csv(train_datafile)
raw_test <- read.csv(test_datafile)
toc()

# 19622 training, 160 variables,  testing 20 obs
# 19216 empty in tons of things with wind
#summary(training$classe)
#summary(training)

#out of the 160 variables how many of them are 'low' in data
#   or information ???

M1 <- sapply(raw_train, function(x) sum(is.na(x))); M1 <- M1[M1 > 0]
M2 <- sapply(raw_train, function(x) sum(x == "", na.rm=TRUE)); M2 <- M2[M2 > 0]
training_validation <- select(raw_train, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))

testing <- select(raw_test, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))


names(testing)


# now split into a validation set
inTrain = createDataPartition(training_validation$classe, p = 0.7)[[1]]
training = training_validation[ inTrain,]
validation = training_validation[-inTrain,]



tic("linear discriminent")
lda_fit <- train(classe ~ ., data=training, method="lda")
lda_pred <- predict(lda_fit, newdata=validation)
toc()

tic("gradient boosting")
gbm_fit <- train(classe ~ ., data=training, method="gbm", verbose=FALSE)
gbm_pred <- predict(gbm_fit, newdata=validation)
toc()


tic("random forest model")
rf_fit <- train(classe ~ ., data=training, method="rf")
rf_pred <- predict(rf_fit, newdata=validation)
toc()

lda_pred
testing$classe


# note it took at least 30 minutes to run the first one
#   the thid one failed "accuracy metric values missing"

confusionMatrix(rf_pred, lda_pred)$overall[1]
confusionMatrix(gbm_pred, validation$classe)$overall[1]
confusionMatrix(lda_pred, validation$classe)
confusionMatrix(lda_pred, validation$classe)$overall[1]
confusionMatrix(rf_pred, validation$classe)$overall[1]
confusionMatrix(lda_pred, rf_pred)$overall[1]
confusionMatrix(lda_pred, gbm_pred)$overall[1]
confusionMatrix(rf_pred, gbm_pred)$overall[1]




#############################################
#
#  Now we hjave lda_fit.RData, gbm_fit.RData, and 
#   rf_fit.RData all trained
require(caret)

#
load(file="rf_fit.RData")
load(file="gbm_fit.RData")
load(file="lda_fit.RData")

# prediction on the training set
lda_p1 <- predict(lda_fit, newdata=training)
confusionMatrix(lda_p1, training$classe)$overall
confusionMatrix(lda_p1, training$classe)$table


gbm_p1 <- predict(gbm_fit, newdata=training)
confusionMatrix(gbm_p1, training$classe)$overall
confusionMatrix(gbm_p1, training$classe)$table

rf_p1 <- predict(rf_fit, newdata=training)
confusionMatrix(rf_p1, training$classe)$overall
confusionMatrix(rf_p1, training$classe)$table


# validation score
rf_p2 <- predict(rf_fit, newdata=validation)
confusionMatrix(rf_p2, validation$classe)$overall
confusionMatrix(rf_p2, validation$classe)$table


gbm_p2 <- predict(gbm_fit, newdata=validation)
confusionMatrix(gbm_p2, validation$classe)$overall
confusionMatrix(gbm_p2, validation$classe)$table

lda_p2 <- predict(lda_fit, newdata=validation)
confusionMatrix(lda_p2, validation$classe)$overall
confusionMatrix(lda_p2, validation$classe)$table


# combined model
#   you build a model that will guess based on the outputs
#   from the three other models
#  the other three model outputs predicted on the training set
combPredTraining <- data.frame(
    rf_pred = predict(rf_fit, newdata=training), 
    gbm_pred = predict(gbm_fit, newdata=training), 
    lda_pred = predict(gbm_fit, newdata=training), 
    classe=training$classe)

# build random forest, 13737 observations, with 3 predictors
comb_fit <- train(classe ~ ., method="rf", data=combPredTraining)

save(file="comb_fit.RData", comb_fit)
load(file="comb_fit.RData")

# for consitency, get a p1 (training) prediction
comb_p1 <- predict(comb_fit, newdata=combPredTraining)
confusionMatrix(comb_p1, combPredTraining$classe)$overall
confusionMatrix(rf_p1, training$classe)$overall
# we are exact match for accuracy



# is it a better fit than original RF (which was pretty good)
comb_p2 <- predict(comb_fit, newdata=data.frame(
    rf_pred = predict(rf_fit, newdata=validation), 
    gbm_pred = predict(gbm_fit, newdata=validation), 
    lda_pred = predict(gbm_fit, newdata=validation)))
confusionMatrix(comb_p2, validation$classe)$overall
confusionMatrix(rf_p2, validation$classe)$overall
# again exactly the same




#
#
#
#   all sorts of other models


# validation score
tic("Naive Bayes") #assumes, training/validation are setup
nb_fit <- train(classe ~ ., data=training, method="nb")
nb_p1 <- predict(nb_fit, newdata=training)
nb_p2 <- predict(nb_fit, newdata=validation)
confusionMatrix(nb_p1, training$classe)$overall
confusionMatrix(nb_p2, validation$classe)$overall
confusionMatrix(nb_p2, validation$classe)$table
toc()
save(nb_fit,file="nb_fit.RData")

# Default Neural Net was about 75% on the training data
tic("Neural Network") #assumes, training/validation are setup
nnet_fit <- train(classe ~ ., data=training, method="nnet")
nnet_p1 <- predict(nb_fit, newdata=training)
nnet_p2 <- predict(nb_fit, newdata=validation)
confusionMatrix(nnet_p1, training$classe)$overall
confusionMatrix(nnet_p2, validation$classe)$overall
confusionMatrix(nnet_p2, validation$classe)$table
toc()

tic("Ranger Random Forest") #assumes, training/validation are setup
ranger_fit <- train(classe ~ ., data=training, method="nnet")
ranger_p1 <- predict(ranger_fit, newdata=training)
ranger_p2 <- predict(ranger_fit, newdata=validation)
confusionMatrix(ranger_p1, training$classe)$overall
confusionMatrix(ranger_p2, validation$classe)$overall
confusionMatrix(ranger_p2, validation$classe)$table
toc()

tic("Neural Network") #assumes, training/validation are setup
nnet_fit <- train(classe ~ ., data=training, method="nnet")
nnet_p1 <- predict(nb_fit, newdata=training)
nnet_p2 <- predict(nb_fit, newdata=validation)
confusionMatrix(nnet_p1, training$classe)$overall
confusionMatrix(nnet_p2, validation$classe)$overall
confusionMatrix(nnet_p2, validation$classe)$table
toc()










##############
# create folds
flds <- createFolds(training_validation$classe, k = 3, list = TRUE, returnTrain = FALSE)
names(flds)[1] <- "train"
flds[[1]]

# then call on training_validation[flds[[k]],  ]  k=0,1,2


tic("Parallel Random Forest") #assumes, training/validation are setup
parRF_fit <- train(classe ~ ., data=training, method="parRF")
ranger_p1 <- predict(ranger_fit, newdata=training)
ranger_p2 <- predict(ranger_fit, newdata=validation)
confusionMatrix(ranger_p1, training$classe)$overall
confusionMatrix(ranger_p2, validation$classe)$overall
confusionMatrix(ranger_p2, validation$classe)$table
toc()





############################
# #
#  some parallel compute notes
require(foreach)
require(doParallel)
registerDoParallel(cores=detectCores())
foreach(i=1:4) %dopar% sqrt(i)




trials <- 10000
x <- iris[which(iris[,5] != "setosa"), c(1,5)]
ptime <- system.time({ 
    r <- foreach(icount(trials), .combine=cbind) %dopar% { 
        ind <- sample(100,100,replace=TRUE)
        result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
        coefficients(result1)}
    })[3]
ptime

# change dopar to do for single processer
stime <- system.time({ 
    r <- foreach(icount(trials), .combine=cbind) %do% { 
        ind <- sample(100,100,replace=TRUE)
        result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
        coefficients(result1)}
})[3]
stime




##################################################
#
#  Attempt #3, 
#
raw_train <- read.csv(train_datafile)
raw_test <- read.csv(test_datafile)
M1 <- sapply(raw_train, function(x) sum(is.na(x))); M1 <- M1[M1 > 0]
M2 <- sapply(raw_train, function(x) sum(x == "", na.rm=TRUE)); M2 <- M2[M2 > 0]
training_validation <- select(raw_train, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))

testing <- select(raw_test, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))
inTrain = createDataPartition(training_validation$classe, p = 0.9)[[1]]
training = training_validation[ inTrain,]
validation = training_validation[-inTrain,]
flds <- createFolds(training$classe, k = 3, list = TRUE, returnTrain = FALSE)


tic("rf model1")
rf_fit1 <- train(classe ~ ., data=training[-flds[[1]], ], method="parRF")
rf_pred1 <- predict(rf_fit1, newdata=training[ flds[[1]],])
toc()
save(rf_fit1, file="rf_fit1.RData")

tic("rf model2")
rf_fit2 <- train(classe ~ ., data=training[-flds[[2]], ], method="parRF")
rf_pred2 <- predict(rf_fit2, newdata=training[ flds[[2]],])
toc()
save(rf_fit2, file="rf_fit2.RData")


tic("rf model3")
rf_fit3 <- train(classe ~ ., data=training[-flds[[3]], ], method="parRF")
rf_pred3 <- predict(rf_fit3, newdata=training[ flds[[3]],])
toc()
save(rf_fit3, file="rf_fit3.RData")



confusionMatrix(rf_pred1, training[ flds[[1]],]$classe)
confusionMatrix(rf_pred2, training[ flds[[2]],]$classe)
confusionMatrix(rf_pred3, training[ flds[[3]],]$classe)

combined_data <- data.frame(
    rf1_pred = predict(rf_fit1, newdata=training), 
    rf2_pred = predict(rf_fit2, newdata=training), 
    rf3_pred = predict(rf_fit3, newdata=training), 
    classe = training$classe)

tic("fit combined model")
combined_fit <- train(classe ~ ., data=combined_data, method="parRF")
toc()
save(combined_fit, file="combined_fit.RData")

combined_validation <- data.frame(
    rf1_pred = predict(rf_fit1, newdata=validation), 
    rf2_pred = predict(rf_fit2, newdata=validation), 
    rf3_pred = predict(rf_fit3, newdata=validation)
)
validation_pred <- predict(combined_fit, newdata=combined_validation)
confusionMatrix(validation_pred, validation$classe)






# combine the confusionMatrix bounds