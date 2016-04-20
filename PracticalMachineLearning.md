# Practical Machine Learning - Course Project
Vhishma Pant  

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## Data Preparation


```r
# Load required libraries
library(caret)
library(ggplot2)
library(randomForest)
```

#### Data Source
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

We will need to download those files and place in the working directory.


```r
trainFile <- "pml-training.csv"
testFile <- "pml-testing.csv"

train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists(trainFile)) {
  download.file(url=train_url, destfile=trainFile)
}

test_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists(testFile)) {
  download.file(url=test_url, destfile=testFile)
}
```


#### Loading the Data


```r
train <- read.csv(trainFile, na.strings=c("NA",""), header=TRUE)
test <- read.csv(testFile, na.strings=c("NA",""), header=TRUE)
```

#### Inspect the Data


```r
#head(train)
#summary(train)
#str(train)
```


```r
#head(test)
#summary(test)
#str(test)
```

We can see that we have training data with 160 variables and many missing values. SO, we will remove variables with near zero variance, variables with mostly missing data, and variables that are obviously not useful as predictors.

#### Remove the variables with low variance
We will analyze and omit the variables with zero variance as those would not contribute to the modeling process.


```r
#low_var <- nearZeroVar(train, saveMetrics=TRUE)
#train <- train[-low_var, ]
```


```r
#dim(train)
```

We have reduced the number of variables to 100.

#### Eliminate the variables with missing or null values

The variables with data that is predominantly missing are eliminated. Now, there are only 59 variables remaining.

```r
na_count <- summary(is.na(train))
na_count1 = sapply(train, function(x) {sum(is.na(x))})
cols_with_nas = names(na_count1[na_count1>18000])
train = train[, !names(train) %in% cols_with_nas]
dim(train)
```

```
## [1] 19622    60
```

#### Remove unuseful variables
The first 6 variables are removed as they are not useful. They contain descriptive information that would not be used in analysis. As can be seen, 53 variables now remain out of an original 160 variables.

```r
train <- train[-c(1:6)]
dim(train)
```

```
## [1] 19622    54
```

### Splitting training/testing data
We will be splitting the data for the validation process. We will now have two data sets: 60% for myTraining, 40% for myTesting. This is done for the model to be validated against a clean dataset.


```r
inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
myTraining <- train[inTrain, ]
myTesting <- train[-inTrain, ]

dim(myTraining)
```

```
## [1] 11776    54
```

```r
dim(myTesting)
```

```
## [1] 7846   54
```

## Modeling

### Random Forest Model
I will be using random-forest technique to generate the predictive model.

```r
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=myTraining, method="rf", trControl=fitControl)
```


```r
# print final model to see tuning parameters it chose
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.31%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    0    0    1 0.0005973716
## B   11 2265    2    1    0 0.0061430452
## C    0    6 2048    0    0 0.0029211295
## D    0    0    7 1922    1 0.0041450777
## E    0    0    0    6 2159 0.0027713626
```

Now, the final model is used to predict the outcome in the validation data set. This will be used to do cross validation on myTesting data.

```r
pred_test <- predict(fit, myTesting)
confusionMatrix(myTesting$classe, pred_test)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    3 1515    0    0    0
##          C    0    4 1362    2    0
##          D    0    0    0 1286    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2849          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9974   1.0000   0.9984   1.0000
## Specificity            1.0000   0.9995   0.9991   1.0000   1.0000
## Pos Pred Value         1.0000   0.9980   0.9956   1.0000   1.0000
## Neg Pred Value         0.9995   0.9994   1.0000   0.9997   1.0000
## Prevalence             0.2849   0.1936   0.1736   0.1642   0.1838
## Detection Rate         0.2845   0.1931   0.1736   0.1639   0.1838
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9993   0.9984   0.9995   0.9992   1.0000
```


```r
pred_train <- predict(fit, myTraining)
confusionMatrix(myTraining$classe, pred_train)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    2 2274    3    0    0
##          C    0    4 2049    1    0
##          D    0    0    2 1928    0
##          E    0    2    0    2 2161
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9978, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9983          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9976   0.9984   1.0000
## Specificity            1.0000   0.9995   0.9995   0.9998   0.9996
## Pos Pred Value         1.0000   0.9978   0.9976   0.9990   0.9982
## Neg Pred Value         0.9998   0.9994   0.9995   0.9997   1.0000
## Prevalence             0.2845   0.1936   0.1744   0.1640   0.1835
## Detection Rate         0.2843   0.1931   0.1740   0.1637   0.1835
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9997   0.9984   0.9985   0.9991   0.9998
```

From the model summary, we can see that during the cross validation we get accuracy of 98.8% with out-of-sample error of 0.02%.

### Model re-run

```r
predict_final <- predict(fit, test)
print(predict_final)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

