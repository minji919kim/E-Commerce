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

df <- read_csv("Online_Shoppers_Oversampled.csv")
options(scipen=10)

df <- dummy_cols(df, select_columns = c('Revenue'), remove_first_dummy = TRUE) %>% select(-Revenue)
df$Revenue_TRUE = as.factor(df$Revenue_TRUE)

dftrain <- filter(df,partition == 'train') %>% select(-partition)
dftest <- filter(df,partition == 'test') %>% select(-partition)


# k-fold Tree -------------------------------------------------------------

set.seed(1)
kfoldtree <- train(Revenue_TRUE~., data=dftrain, method="rpart",
                   trControl=trainControl("cv",number=10), tuneLength=8)
kfoldtree


# cp = 0.001096792
finaltree <- rpart(Revenue_TRUE~., data=dftrain, method="class",cp=0.001096792)
summary(finaltree)
rpart.plot(finaltree,roundint=FALSE,nn=TRUE,extra=4)

ktree_train <- mutate(dftrain, Prediction = predict(finaltree, dftrain,type="class"))
mean(~(Revenue_TRUE == Prediction), data=ktree_train)
ktree_train_roc <- roc(as.numeric(ktree_train$Revenue_TRUE), as.numeric(ktree_train$Prediction))
auc(ktree_train_roc)

tally(Revenue_TRUE ~ Prediction, data = ktree_train) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=ktree_train) %>% prop.table(margin=1) %>% round(2)


ktree_test <- mutate(dftest, Prediction = predict(finaltree, dftest,type="class"))
mean(~(Revenue_TRUE == Prediction), data=ktree_test)

tally(Revenue_TRUE ~ Prediction, data = ktree_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=ktree_test) %>% prop.table(margin=1) %>% round(2)
ktree_test_roc <- roc(as.numeric(ktree_test$Revenue_TRUE), as.numeric(ktree_test$Prediction))
auc(ktree_test_roc)


# k-fold KNN --------------------------------------------------------------

# KNN
dfdum <- dummy_cols(df, 
                    select_columns = c('Month','VisitorType','Weekend','SpecialDay',
                                       'OperatingSystems','Region', 'TrafficType','Browser'), 
                    remove_first_dummy = TRUE)
dfdum <- select(dfdum,-Month,-VisitorType,-Weekend,-Region,-TrafficType,-OperatingSystems,-Browser,-SpecialDay)


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

dfknnkfold_train <- select(dfnormdumtrain,Revenue_TRUE,PageValues,Month_Nov,ExitRates,ProductRelated_Duration,Month_May,
                           TrafficType_5,TrafficType_13,Region_5,VisitorType_Other,Month_Feb,Month_Mar,Month_Dec,
                           TrafficType_8,Browser_6,TrafficType_2, Administrative,Browser_3,TrafficType_11,Region_2,
                           SpecialDay_0.6,TrafficType_4, TrafficType_10,TrafficType_20,OperatingSystems_2,Browser_2,
                           Region_9,OperatingSystems_7,Informational,SpecialDay_0.2,OperatingSystems_4,Month_Oct,Weekend_TRUE)

set.seed(1)
kfold_knn <- train(Revenue_TRUE~., data=dfknnkfold_train, method="knn",
                      trControl=trainControl("cv",number=10), tuneGrid=expand.grid(k = 1:19))
kfold_knn

# K=1
knnkfold_train <- knn(dfknntrain, dfknntrain, dfnormdumtrain$Revenue_TRUE, k = 1)
knnkfold_pred_train <- mutate(dfnormdumtrain, Prediction = knnkfold_train) 
mean(~(Revenue_TRUE == Prediction), data=knnkfold_pred_train)
knn_train_roc <- roc(as.numeric(knnkfold_pred_train$Revenue_TRUE), as.numeric(knnkfold_pred_train$Prediction))
auc(knn_train_roc)
tally(Revenue_TRUE ~ Prediction, data=knnkfold_pred_train) %>% prop.table(margin=1) %>% round(2)

# test
knnkfold_test <- knn(dfknntrain, dfknntest, dfnormdumtrain$Revenue_TRUE, k = 1) 
knnkfold_pred_test <- mutate(dfnormdumtest, Prediction = knnkfold_test)
mean(~(Revenue_TRUE == Prediction),data=knnkfold_pred_test)
knn_test_roc <- roc(as.numeric(knnkfold_pred_test$Revenue_TRUE), as.numeric(knnkfold_pred_test$Prediction))
auc(knn_test_roc)

# Tally table
tally(Revenue_TRUE ~ Prediction, data = knnkfold_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=knnkfold_pred_test) %>% prop.table(margin=1) %>% round(2)


# k-fold NN ---------------------------------------------------------------

# NN

set.seed(1)
kfold_nn <- train(Revenue_TRUE~., data=dfnormdumtrain, method="nnet",
                   trControl=trainControl("cv",number=10), tuneGrid=expand.grid(size = 1:13,decay=0.01))
kfold_nn

set.seed(1)
nn12 <- nnet(Revenue_TRUE~., data = dfnormdumtrain, size=12,linout=F,decay=0.01)
nn12_pred_train <- mutate(dfnormdumtrain, Prediction = predict(nn12, dfnormdumtrain) %>% round())
mean(~(Revenue_TRUE == Prediction),data=nn12_pred_train)
nn12_train_roc <- roc(nn12_pred_train$Revenue_TRUE, nn12_pred_train$Prediction)
auc(nn12_train_roc)
tally(Revenue_TRUE ~ Prediction, data=nn12_pred_train) %>% prop.table(margin=1) %>% round(2)

# use nn12 to test
nn12_pred_test <- mutate(dfnormdumtest, Prediction = predict(nn12, dfnormdumtest) %>% round()) 
mean(~(Revenue_TRUE == Prediction), data=nn12_pred_test)
nn12_test_roc <- roc(nn12_pred_test$Revenue_TRUE, nn12_pred_test$Prediction)
auc(nn12_test_roc)

tally(Revenue_TRUE ~ Prediction, data = nn12_pred_test) %>% addmargins()
tally(Revenue_TRUE ~ Prediction, data=nn12_pred_test) %>% prop.table(margin=1) %>% round(2)
