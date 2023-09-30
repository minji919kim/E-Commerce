library(dplyr)
library(haven)
library(caret)
library(fastDummies)
library(rpart)
library(rpart.plot)
library(FNN)
library(nnet)
library(mosaic)
library(e1071)
library(tidyverse)
library(pROC)
library(gains)
library(caTools)
library(randomForest)

df <- read_csv('Online_Shoppers_Oversampled.csv')


df <- dummy_cols(df, select_columns = c('Revenue'), remove_first_dummy = TRUE) %>% select(-Revenue)
df$Revenue_TRUE = as.factor(df$Revenue_TRUE)

dftrain <- filter(df,partition == 'train') %>% select(-partition)
dftest <- filter(df,partition == 'test') %>% select(-partition)

rf <- randomForest(Revenue_TRUE~.,data = dftrain)
trainpreds <- mutate(dftrain, Prediction = predict(rf, newdata = dftrain, type= "class"))
mean(~(Revenue_TRUE == Prediction), data = trainpreds)
train_roc <- roc(as.numeric(trainpreds$Revenue_TRUE), as.numeric(trainpreds$Prediction))
auc(train_roc)

tally(Revenue_TRUE ~ Prediction, data=trainpreds) %>%  prop.table(margin=1) %>% round(2)

testpreds <- mutate(dftest, Prediction = predict(rf, newdata = dftest, type= "class"))
mean(~(Revenue_TRUE == Prediction), data = testpreds)
test_roc <- roc(as.numeric(testpreds$Revenue_TRUE), as.numeric(testpreds$Prediction))
auc(test_roc)

tally(Revenue_TRUE ~ Prediction, data=testpreds) %>%  prop.table(margin=1) %>% round(2)

