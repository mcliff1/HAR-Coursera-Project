# Week 4 ProjectNotes
#  Practical Machine Learning


setwd("H:/Programming/Coursera/Machine Learning/project/")
test_datafile <- "./data/pml-training.csv"
train_datafile <- "./data/pml-testing.csv"
train_fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
dir.create("data", showWarnings=FALSE)
if(!file.exists(train_datafile)) {
    download.file(train_fileURL, destfile=train_datafile)
}
if(!file.exists(test_datafile)) {
    download.file(test_fileURL, destfile=test_datafile)
}
training <- read.csv(train_datafile)
testing <- read.csv(test_datafile)

# 19622 training, 160 variables,  testing 20 obs
summary(training$classe)
summary(training)

#out of the 160 variables how many of them are 'low' in data
#   or information ???


require(caret)
rf_fit <- train(classe ~ ., data=training, method="rf", na.action=na.exclude)
rf_pred <- predict(rf_fit, newdata=testing)

gbm_fit <- train(classe ~ ., data=training, method="gbm", verbose=FALSE, na.action=na.exclude)
gbm_pred <- predict(gbm_fit, newdata=testing)

lda_fit <- train(classe ~ ., data=training, method="lda", na.action=na.exclude)
lda_pred <- predict(lda_fit, newdata=testing)