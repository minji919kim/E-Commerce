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

df_old <- read_csv("online_shoppers_partition.csv")

set.seed(1)
df_TRUE <- filter(df_old,Revenue==TRUE)
df_FALSE <- filter(df_old,Revenue==FALSE)
df_overtrue <- sample(df_TRUE,nrow(df_FALSE),replace=TRUE) %>% select(-orig.id)
df <- rbind(df_overtrue, df_FALSE)

write.csv(df,"C:/Users/dujing/Documents/Study/Boston College/Fall 2022/ML for BI/Online_Shoppers_Oversampled.csv",row.names=FALSE)

df <- read_csv("Online_Shoppers_Oversampled.csv")

df <- dummy_cols(df, select_columns = c('Revenue'), remove_first_dummy = TRUE) %>% select(-Revenue)

# Logistic Regression -------------------------------------------------------

dfdum <- dummy_cols(df, 
                    select_columns = c('Month','VisitorType','Weekend','SpecialDay',
                                       'OperatingSystems','Region', 'TrafficType','Browser'), 
                    remove_first_dummy = TRUE)
dfdum <- select(dfdum,-Month,-VisitorType,-Weekend,-Region,-TrafficType,-OperatingSystems,-Browser,-SpecialDay)

dfdumtrain <- filter(dfdum, partition == 'train') %>% select(-partition)
dfdumtest <- filter(dfdum, partition == 'test') %>% select(-partition)


# Stepwise
nullglm <- glm(Revenue_TRUE~1,data=dfdumtrain, family=binomial)
summary(nullglm)
allmodel <-glm(Revenue_TRUE ~., data = dfdumtrain, family = binomial)
stepglm <- step(nullglm,scope=formula(allmodel))
summary(stepglm)
step_pred_train <- mutate(dfdumtrain, Prediction = predict(stepglm, dfdumtrain, type='response') %>% round())
mean(~(Revenue_TRUE == Prediction), data=step_pred_train)
step_train_roc <- roc(step_pred_train$Revenue_TRUE, step_pred_train$Prediction)
auc(step_train_roc)
tally(Revenue_TRUE ~ Prediction, data=step_pred_train) %>% prop.table(margin=1) %>% round(2)

# Test in test data
step_pred_test <- mutate(dfdumtest, Prediction = predict(stepglm, dfdumtest, type='response') %>% round())
mean(~(Revenue_TRUE == Prediction), data=step_pred_test)
step_test_roc <- roc(step_pred_test$Revenue_TRUE, step_pred_test$Prediction)
auc(step_test_roc)

# tally table
tally(Revenue_TRUE ~ Prediction, data = step_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=step_pred_test) %>% prop.table(margin=1) %>% round(2)

# Classification Tree ---------------------------------------------------------


# filter partition without dummies
dftrain <- filter(df,partition == 'train') %>% select(-partition)
dftest <- filter(df,partition == 'test') %>% select(-partition)

# Tree03
tree03 <- rpart(Revenue_TRUE ~., data=dftrain, method="class", cp=0.03)
summary(tree03)
rpart.plot(tree03,roundint=FALSE,nn=TRUE,extra=4)

tree03_train <- mutate(dftrain, Prediction = predict(tree03, dftrain,type="class"))
mean(~(Revenue_TRUE == Prediction), data=tree03_train)
tree03_train_roc <- roc(as.numeric(tree03_train$Revenue_TRUE), as.numeric(tree03_train$Prediction))
auc(tree03_train_roc)
tally(Revenue_TRUE ~ Prediction, data=tree03_train) %>% prop.table(margin=1) %>% round(2)


# use tree03 to test
tree03_test <- mutate(dftest, Prediction = predict(tree03, dftest,type="class"))
mean(~(Revenue_TRUE == Prediction), data=tree03_test)
tree03_test_roc <- roc(as.numeric(tree03_test$Revenue_TRUE), as.numeric(tree03_test$Prediction))
auc(tree03_test_roc)

tally(Revenue_TRUE ~ Prediction, data = tree03_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=tree03_test) %>% prop.table(margin=1) %>% round(2)

tree_output = data.frame()
for (num in seq(0.005,0.03,0.001)) {
  treenum <- rpart(Revenue_TRUE ~., data=dftrain, method="class", cp=num)
  treenum_train <- mutate(dftrain, Prediction = predict(treenum, dftrain,type="class"))
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), data=treenum_train) 
  tree_roc <- roc(as.numeric(treenum_train$Revenue_TRUE), as.numeric(treenum_train$Prediction))
  area <- auc(tree_roc)
  
  output = c(num, percentage_correct, area)
  tree_output = rbind(tree_output, output)
}    
colnames(tree_output) <- c("cp", "Training %Correct")



# KNN ---------------------------------------------------------------------

# Use data with dummies
# rescale
dfnorm <- mutate(dfdum, 
                 Administrative=(Administrative-min(Administrative))/(max(Administrative)-min(Administrative)),
                 Administrative_Duration = (Administrative_Duration-min(Administrative_Duration))/(max(Administrative_Duration)-min(Administrative_Duration)),
                 Informational = (Informational-min(Informational))/(max(Informational)-min(Informational)),
                 Informational_Duration = (Informational_Duration-min(Informational_Duration))/(max(Informational_Duration)-min(Informational_Duration)),
                 ProductRelated = (ProductRelated-min(ProductRelated))/(max(ProductRelated)-min(ProductRelated)),
                 ProductRelated_Duration = (ProductRelated_Duration-min(ProductRelated_Duration))/(max(ProductRelated_Duration)-min(ProductRelated_Duration)),
                 PageValues = (PageValues-min(PageValues))/(max(PageValues)-min(PageValues)),
                 BounceRates = (BounceRates-min(BounceRates))/(max(BounceRates)-min(BounceRates)),
                 ExitRates = (ExitRates-min(ExitRates))/(max(ExitRates)-min(ExitRates)))

# partition
dfnormdumtrain <- filter(dfnorm,partition=='train') %>% select(-partition)
dfnormdumtest <- filter(dfnorm,partition=='test') %>% select(-partition)

# Select KNN variables(Marginally Significant Variables with p <= 0.05)
dfknntrain <- select(dfnormdumtrain,PageValues,Month_Nov,ExitRates,ProductRelated_Duration,Month_May,
                     TrafficType_5,TrafficType_13,Region_5,VisitorType_Other,Month_Feb,Month_Mar,Month_Dec,
                     TrafficType_8,Browser_6,TrafficType_2, Administrative,Browser_3,TrafficType_11,Region_2,
                     SpecialDay_0.6,TrafficType_4, TrafficType_10,TrafficType_20,OperatingSystems_2,Browser_2,
                     Region_9,OperatingSystems_7,Informational,SpecialDay_0.2,OperatingSystems_4,Month_Oct,Weekend_TRUE)
dfknntest <- select(dfnormdumtest,PageValues,Month_Nov,ExitRates,ProductRelated_Duration,Month_May,
                    TrafficType_5,TrafficType_13,Region_5,VisitorType_Other,Month_Feb,Month_Mar,Month_Dec,
                    TrafficType_8,Browser_6,TrafficType_2, Administrative,Browser_3,TrafficType_11,Region_2,
                    SpecialDay_0.6,TrafficType_4, TrafficType_10,TrafficType_20,OperatingSystems_2,Browser_2,
                    Region_9,OperatingSystems_7,Informational,SpecialDay_0.2,OperatingSystems_4,Month_Oct,Weekend_TRUE)

# KNN3
knn3_train <- knn(dfknntrain, dfknntrain, dfnormdumtrain$Revenue_TRUE, k = 3)
knn3_pred_train <- mutate(dfnormdumtrain, Prediction = knn3_train) 
mean(~(Revenue_TRUE == Prediction), data=knn3_pred_train)
knn3_train_roc <- roc(as.numeric(knn3_pred_train$Revenue_TRUE), as.numeric(knn3_pred_train$Prediction))
auc(knn3_train_roc)
tally(Revenue_TRUE ~ Prediction, data=knn3_pred_train) %>% prop.table(margin=1) %>% round(2)

# use KNN3 to test
knn3_test <- knn(dfknntrain, dfknntest, dfnormdumtrain$Revenue_TRUE, k = 3) 
knn3_pred_test <- mutate(dfnormdumtest, Prediction = knn3_test)
mean(~(Revenue_TRUE == Prediction),data=knn3_pred_test)
knn3_test_roc <- roc(as.numeric(knn3_pred_test$Revenue_TRUE), as.numeric(knn3_pred_test$Prediction))
auc(knn3_test_roc)

# Tally table
tally(Revenue_TRUE ~ Prediction, data = knn3_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=knn3_pred_test) %>% prop.table(margin=1) %>% round(2)

knn_output = data.frame()
for (num in 2:19) {
  knn_train <- knn(dfknntrain, dfknntrain, dfnormdumtrain$Revenue_TRUE, k = num)
  knn_pred_train <- mutate(dfnormdumtrain, Prediction = knn_train) 
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), data=knn_pred_train) 
  knn_roc <- roc(as.numeric(knn_pred_train$Revenue_TRUE), as.numeric(knn_pred_train$Prediction))
  area <- auc(knn_roc)
  
  output = c(num, percentage_correct,area)
  knn_output = rbind(knn_output, output)
}    
colnames(knn_output) <- c("Number of Neighbors", "Training %Correct",'AUC')

# Neural Networks ---------------------------------------------------------

# Use dfnormdumtrain / dfnormdumtest

# nn11
set.seed(1)
nn11 <- nnet(Revenue_TRUE~., data = dfnormdumtrain, size=11,linout=F,decay=0.01)
nn11_pred_train <- mutate(dfnormdumtrain, Prediction = predict(nn11, dfnormdumtrain) %>% round())
mean(~(Revenue_TRUE == Prediction),data=nn11_pred_train)
nn11_train_roc <- roc(nn11_pred_train$Revenue_TRUE, nn11_pred_train$Prediction)
auc(nn11_train_roc)
tally(Revenue_TRUE ~ Prediction, data=nn11_pred_train) %>% prop.table(margin=1) %>% round(2)

# use nn11 to test
nn11_pred_test <- mutate(dfnormdumtest, Prediction = predict(nn11, dfnormdumtest) %>% round()) 
mean(~(Revenue_TRUE == Prediction), data=nn11_pred_test)
nn11_test_roc <- roc(nn11_pred_test$Revenue_TRUE, nn11_pred_test$Prediction)
auc(nn11_test_roc)

tally(Revenue_TRUE ~ Prediction, data = nn11_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=nn11_pred_test) %>% prop.table(margin=1) %>% round(2)

nn_output = data.frame()
for (num in 2:13) {
  set.seed(1)
  nn <- nnet(Revenue_TRUE~., data = dfnormdumtrain, size=num,linout=F,decay=0.01)
  nn_pred_train <- mutate(dfnormdumtrain, Prediction = predict(nn, dfnormdumtrain) %>% round())
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), data=nn_pred_train) 
  
  output = c(num, percentage_correct)
  nn_output = rbind(nn_output, output)
}    
colnames(knn_output) <- c("Number of Hidden Nodes", "Training %Correct")



# Naive Bayes -------------------------------------------------------------
dfdumbin <- select(dfdum,-Administrative,-Administrative_Duration,-Informational,
                   -Informational_Duration,-ProductRelated,-ProductRelated_Duration,-ExitRates,
                   -BounceRates,-PageValues)
dfdumbinfactor <- lapply(dfdumbin,as.factor) %>% data.frame()

dfdumbinfactortrain <- filter(dfdumbinfactor, partition == 'train') %>% select(-partition)
dfdumbinfactortest <- filter(dfdumbinfactor, partition == 'test') %>% select(-partition)

# Naive Bayes
nb <- naiveBayes(Revenue_TRUE~. ,data=dfdumbinfactortrain)
nb_pred_train <- mutate(dfdumbinfactortrain, Prediction = predict(nb, dfdumbinfactortrain,type="class"))
mean(~(Revenue_TRUE == Prediction), data=nb_pred_train)
nb_train_roc <- roc(as.numeric(nb_pred_train$Revenue_TRUE), as.numeric(nb_pred_train$Prediction))
auc(nb_train_roc)
tally(Revenue_TRUE ~ Prediction, data=nb_pred_train) %>% prop.table(margin=1) %>% round(2)

# Test
nb_pred_test <- mutate(dfdumbinfactortest, Prediction = predict(nb, dfdumbinfactortest,type="class"))
mean(~(Revenue_TRUE == Prediction), data=nb_pred_test)
nb_test_roc <- roc(as.numeric(nb_pred_test$Revenue_TRUE), as.numeric(nb_pred_test$Prediction))
auc(nb_test_roc)

tally(Revenue_TRUE ~ Prediction, data = nb_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=nb_pred_test) %>% prop.table(margin=1) %>% round(2)


# SVM ---------------------------------------------------------------------

# svm29
svm29 <- svm(Revenue_TRUE ~., data=dfnormdumtrain, type="C-classification",cost=29,scale=FALSE)
svm29_pred_train <- mutate(dfnormdumtrain, Prediction = predict(svm29, dfnormdumtrain)) 
mean(~(Revenue_TRUE == Prediction), data=svm29_pred_train)
svm_train_roc <- roc(as.numeric(svm29_pred_train$Revenue_TRUE), as.numeric(svm29_pred_train$Prediction))
auc(svm_train_roc)
tally(Revenue_TRUE ~ Prediction, data=svm29_pred_train) %>% prop.table(margin=1) %>% round(2)

# Test
svm29_pred_test <- mutate(dfnormdumtest, Prediction = predict(svm29, dfnormdumtest)) 
mean(~(Revenue_TRUE == Prediction), data=svm29_pred_test)
svm_test_roc <- roc(as.numeric(svm29_pred_test$Revenue_TRUE), as.numeric(svm29_pred_test$Prediction))
auc(svm_test_roc)

tally(Revenue_TRUE ~ Prediction, data = svm29_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=svm29_pred_test) %>% prop.table(margin=1) %>% round(2)

svm_output = data.frame()
for (num in seq(5,80,5)) {
  svm <- svm(Revenue_TRUE ~., data=dfnormdumtrain,type="C-classification",cost=num,scale=FALSE)
  svm_pred_train <- mutate(dfnormdumtrain, Prediction = predict(svm, dfnormdumtrain))
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), 
                             data=svm_pred_train) 
  output = c(num, percentage_correct)
  svm_output = rbind(svm_output, output)
}    

colnames(svm_output) <- c("Cost", "Percentage Correct")
