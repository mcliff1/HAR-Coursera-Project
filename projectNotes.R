# Week 4 ProjectNotes
#  Practical Machine Learning


setwd("H:/Programming/Coursera/Machine Learning/project/")
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

M1 <- sapply(raw_test, function(x) sum(is.na(x))); M1 <- M1[M1 > 0]
M2 <- sapply(raw_test, function(x) sum(x == "", na.rm=TRUE)); M2 <- M2[M2 > 0]
training <- select(raw_train, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))

testing <- select(raw_test, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
    select( -one_of(names(M1))) %>%
    select( -one_of(names(M2)))


summary(training$new_window)
names(training)

require(caret)
require(ElemStatLearn)
require(pgmm)
require(rpart)
require(gbm)
require(lubridate)
require(forecast)
require(e1071)
rf_fit <- train(classe ~ ., data=training, method="rf")
rf_pred <- predict(rf_fit, newdata=testing)

gbm_fit <- train(classe ~ ., data=training, method="gbm", verbose=FALSE, na.action=na.exclude)
gbm_pred <- predict(gbm_fit, newdata=testing)

lda_fit <- train(classe ~ ., data=training, method="lda", na.action=na.exclude)
lda_pred <- predict(lda_fit, newdata=testing)

# note it took at least 30 minutes to run the first one
#   the thid one failed "accuracy metric values missing"

# this cleans out the things that are na
M <- (sapply(training, function(x) sum(is.na(x)))); M[M>0]
length(M[M>0])
t2 <- select(training, -one_of(names(M[M>0])))
t2

M2 <- sapply(t2, function(x) sum(x == ""))
length(M2[M2>0])
M2[M2>0]

t3 <- select(t2, -one_of(names(M2[M2>0])))

summary(t3)


# we don't want the number, or timestamps either
#  not sure what 'new_window' is
t4 <- select(t3, -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
             
rf_fit <- train(classe ~ ., data=t4, method="rf", na.action=na.exclude)

testing4 <- select(testing, -one_of(names(M[M>0]))) %>%
    select( -one_of(names(M2[M2>0]))) %>%
    select( -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))

rf_pred <- predict(rf_fit, newdata=testing4)


gbm_fit <- train(classe ~ ., data=t4, method="gbm", verbose=FALSE)
gbm_pred <- predict(gbm_fit, newdata=testing4)
gbm_pred

lda_fit <- train(classe ~ ., data=t4, method="lda")

lda_pred <- predict(lda_fit, newdata=testing4)
lda_pred

confusionMatrix(rf_pred, lda_pred)$overall[1]
confusionMatrix(gbm_fit, testing$classe)$overall[1]
confusionMatrix(lda_pred, testing$diagnosis)$overall[1]

confusionMatrix(combPred, testing$diagnosis)$overall[1]