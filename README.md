# Background
E-commerce is on the rise, with many companies choosing online platforms over traditional stores. Despite the large market opportunity, there is much more intense competition among retailers due to unlimited access to different websites. To stay competitive, companies have to understand their customers' preferences and behaviors by collecting and analyzing online shopping data. This information can be crucial for them to devise effective marketing strategies or focus on their targeted customers. Therefore, the e-commerce dataset will be used in this project to predict customer purchasing trends. 

# Objective
Machine learning will be employed to build a predictive model that uses a variety of attributes to see which ones affect their purchasing intent. The model can guide marketing choices to boost sales conversions. For example, if time spent on product pages influences buying decisions, businesses need to strategize to keep customers engaged longer. Two objectives we focus on are: 
Which model has the highest accuracy in predicting customers’ behavior? 
What are the relatively important attributes contributing to customers’ buying decisions? 

# Dataset 
The dataset comes from Kaggle https://www.kaggle.com/datasets/henrysue/online-shoppers-intention, titled “Online Shoppers Intention UCI Machine Learning.”. The dataset contains 12,330 sessions, with each session representing a different website user in a one-year period. Among them, 10,422 sessions did not have a purchase, whereas 1,908 sessions did involve a purchase. There are 18 variables in total, including ten numeric variables (Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay) and eight categorical variables (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend. Revenue). The dependent variable we try to predict is Revenue, which represents whether or not the customers complete the purchase. 

# Data Preprocessing
Some numerical variables have skewed distribution and outliers, so normalization is applied to maintain training stability and performance of the model. There is one numeric variable, named SpecialDay, converted to dummy variables. SpecialDay represents the relative closeness of the browsing date to shopping periods surrounding special days or holidays. SpecialDay only contains six distinct values (0, 0.2, 0.4, 0.6, 0.8, 1), and we think there would be inconsistent differences among the six different values. So, we treat it as categorical and change it into a dummy variable for analysis. The dataset is split into 70% training and 30% testing, 

# Model Selection 
The selected models are regression, decision tree, k-nearest neighbors (KNN), Naive Bayes, support vector machine (SVM), and random forest. The models are run with both original and oversampled data as the dataset is highly imbalanced with around 85% of non-buyers and 15% of buyers. Also, Oversampling seems reasonable since predicting the purchasers' behaviors is more valuable from a business perspective than non-purchasers'. Through the process, our recommended model is the oversampled k-fold classification tree with a complexity parameter of 0.001096792 given test accuracy, rare event accuracy, and AUC. The test accuracy, rare event accuracy, and AUC are 84.16%, 82%, and 84.09%, respectively. The most prominent variables where splits occur are PageValues, Months, ProductRelated Duration, Informational Duration, ExitRates, BounceRates, and Administrative. 

To learn more about the data processing and result, please check them in [data-presentation folder](/data-presentation/)
