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

df <- read_csv("online_shoppers_intention.csv")
options(scipen=10)

partition <- sample(c('train','test'), size = nrow(df), replace = TRUE, prob = c(0.7,0.3))
df <- mutate(df, partition)
write.csv(df,"C:/Users/dujing/Documents/Study/Boston College/Fall 2022/ML for BI/Online_Shoppers_Partition.csv",row.names=FALSE)

df <- read_csv('Online_Shoppers_Partition.csv')
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

# Test in test data
step_pred_test <- mutate(dfdumtest, Prediction = predict(stepglm, dfdumtest, type='response') %>% round())
mean(~(Revenue_TRUE == Prediction), data=step_pred_test)

# tally table
tally(Revenue_TRUE ~ Prediction, data = step_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=step_pred_test) %>% prop.table(margin=1) %>% round(2)

# Classification Tree ---------------------------------------------------------



# filter partition without dummies
dftrain <- filter(df,partition == 'train') %>% select(-partition)
dftest <- filter(df,partition == 'test') %>% select(-partition)

# Tree03
tree03 <- rpart(Revenue_TRUE ~., data=dftrain, method="class", cp=0.03)
rpart.plot(tree03,roundint=FALSE,nn=TRUE,extra=4)

tree03_train <- mutate(dftrain, Prediction = predict(tree03, dftrain,type="class"))
mean(~(Revenue_TRUE == Prediction), data=tree03_train)
tally(Revenue_TRUE ~ Prediction, data=tree03_train) %>% prop.table(margin=1) %>% round(2)

# use tree03 to test
tree03_test <- mutate(dftest, Prediction = predict(tree03, dftest,type="class"))
mean(~(Revenue_TRUE == Prediction), data=tree03_test)

tally(Revenue_TRUE ~ Prediction, data = tree03_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=tree03_test) %>% prop.table(margin=1) %>% round(2)


tree_output = data.frame()
for (num in seq(0.005,0.03,0.001)) {
  treenum <- rpart(Revenue_TRUE ~., data=dftrain, method="class", cp=num)
  treenum_train <- mutate(dftrain, Prediction = predict(treenum, dftrain,type="class"))
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), data=treenum_train) 
  
  output = c(num, percentage_correct)
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
dfknntrain <- select(dfnormdumtrain,PageValues,ExitRates,Month_Nov,ProductRelated_Duration,
                     TrafficType_13,TrafficType_3,Month_Dec,Month_May,Month_Mar,Month_Feb,TrafficType_8,Weekend_TRUE)
dfknntest <- select(dfnormdumtest,PageValues,ExitRates,Month_Nov,ProductRelated_Duration,
                    TrafficType_13,TrafficType_3,Month_Dec,Month_May,Month_Mar,Month_Feb,TrafficType_8,Weekend_TRUE)

# KNN3
knn3_train <- knn(dfknntrain, dfknntrain, dfnormdumtrain$Revenue_TRUE, k = 3)
knn3_pred_train <- mutate(dfnormdumtrain, Prediction = knn3_train) 
mean(~(Revenue_TRUE == Prediction), data=knn3_pred_train)
tally(Revenue_TRUE ~ Prediction, data=knn3_pred_train) %>% prop.table(margin=1) %>% round(2)

# use KNN3 to test
knn3_test <- knn(dfknntrain, dfknntest, dfnormdumtrain$Revenue_TRUE, k = 3) 
knn3_pred_test <- mutate(dfnormdumtest, Prediction = knn3_test)
mean(~(Revenue_TRUE == Prediction),data=knn3_pred_test)

# Tally table
tally(Revenue_TRUE ~ Prediction, data = knn3_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=knn3_pred_test) %>% prop.table(margin=1) %>% round(2)


knn_output = data.frame()
for (num in 2:19) {
  knn_train <- knn(dfknntrain, dfknntrain, dfnormdumtrain$Revenue_TRUE, k = num)
  knn_pred_train <- mutate(dfnormdumtrain, Prediction = knn_train) 
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), data=knn_pred_train) 
  
  output = c(num, percentage_correct)
  knn_output = rbind(knn_output, output)
}    
colnames(knn_output) <- c("Number of Neighbors", "Training %Correct")

# Neural Networks ---------------------------------------------------------

# Use dfnormdumtrain / dfnormdumtest

# nn11
set.seed(1)
nn11 <- nnet(Revenue_TRUE~., data = dfnormdumtrain, size=11,linout=F,decay=0.01)
nn11_pred_train <- mutate(dfnormdumtrain, Prediction = predict(nn11, dfnormdumtrain) %>% round())
mean(~(Revenue_TRUE == Prediction),data=nn11_pred_train)
tally(Revenue_TRUE ~ Prediction, data=nn11_pred_train) %>% prop.table(margin=1) %>% round(2)

# use nn11 to test
nn11_pred_test <- mutate(dfnormdumtest, Prediction = predict(nn11, dfnormdumtest) %>% round()) 
mean(~(Revenue_TRUE == Prediction), data=nn11_pred_test)

tally(Revenue_TRUE ~ Prediction, data = nn11_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=nn11_pred_test) %>% prop.table(margin=1) %>% round(2)

nn_output = data.frame()
for (num in 11:12) {
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
tally(Revenue_TRUE ~ Prediction, data=nb_pred_train) %>% prop.table(margin=1) %>% round(2)

# Test
nb_pred_test <- mutate(dfdumbinfactortest, Prediction = predict(nb, dfdumbinfactortest,type="class"))
mean(~(Revenue_TRUE == Prediction), data=nb_pred_test)

tally(Revenue_TRUE ~ Prediction, data = nb_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=nb_pred_test) %>% prop.table(margin=1) %>% round(2)


# SVM ---------------------------------------------------------------------

# svm29
svm29 <- svm(Revenue_TRUE ~., data=dfnormdumtrain, type="C-classification",cost=29,scale=FALSE)
svm29_pred_train <- mutate(dfnormdumtrain, Prediction = predict(svm29, dfnormdumtrain)) 
mean(~(Revenue_TRUE == Prediction), data=svm29_pred_train)
tally(Revenue_TRUE ~ Prediction, data=svm29_pred_train) %>% prop.table(margin=1) %>% round(2)

# Test
svm29_pred_test <- mutate(dfnormdumtest, Prediction = predict(svm29, dfnormdumtest)) 
mean(~(Revenue_TRUE == Prediction), data=svm29_pred_test)

tally(Revenue_TRUE ~ Prediction, data = svm29_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=svm29_pred_test) %>% prop.table(margin=1) %>% round(2)

svm_output = data.frame()
for (num in seq(20,30,1)) {
  svm <- svm(Revenue_TRUE ~., data=dfnormdumtrain,type="C-classification",cost=num,scale=FALSE)
  svm_pred_train <- mutate(dfnormdumtrain, Prediction = predict(svm, dfnormdumtrain))
  percentage_correct <- mean(~(Revenue_TRUE == Prediction), 
                             data=svm_pred_train) 
  output = c(num, percentage_correct)
  svm_output = rbind(svm_output, output)
}    

colnames(svm_output) <- c("Cost", "Percentage Correct")
