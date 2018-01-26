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

# 19622 training, 160 variables,  testing 20 obs
# 19216 empty in tons of things with wind
#summary(training$classe)
#summary(training)

#out of the 160 variables how many of them are 'low' in data
#   or information ???

M1 <- sapply(raw_train, function(x) sum(is.na(x))); M1 <- M1[M1 > 0]
M2 <- sapply(raw_train, function(x) sum(x == "", na.rm=TRUE)); M2 <- M2[M2 > 0]
training <- select(raw_train, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))

testing <- select(raw_test, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))


summary(training$new_window)
names(training)

rf_fit <- train(classe ~ ., data=training, method="rf")
rf_pred <- predict(rf_fit, newdata=testing)

gbm_fit <- train(classe ~ ., data=training, method="gbm", verbose=FALSE, na.action=na.exclude)
gbm_pred <- predict(gbm_fit, newdata=testing)

lda_fit <- train(classe ~ ., data=training, method="lda", na.action=na.exclude)
lda_pred <- predict(lda_fit, newdata=testing)

# note it took at least 30 minutes to run the first one
#   the thid one failed "accuracy metric values missing"

confusionMatrix(rf_pred, lda_pred)$overall[1]
confusionMatrix(gbm_pred, testing$classe)$overall[1]
confusionMatrix(lda_pred, testing$diagnosis)$overall[1]

confusionMatrix(combPred, testing$diagnosis)$overall[1]