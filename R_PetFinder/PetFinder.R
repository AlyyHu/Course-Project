#####################################################################################
###                        Final Project
###                        Team 38
###                        PetFinder 
#####################################################################################
install.packages('quantreg')
library(e1071)
library(rpart)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(caret)
library(MLmetrics)
library(GGally)
library(MASS)
library(glmnet)
library(randomForest)
library(nnet)
source("DataAnalyticsFunctions.R")
#####################################################################################
###
###                        Data Understanding
###     
train <- read.csv('train.csv')
summary(train)
names(train)
## Check the independence of three Colors, remove color 2 and color 3 since they are correlated
## with each other

x <- table(train$Color1,train$Color2)
chisq.test(x)

y <- table(train$Color2,train$Color3)
chisq.test(y)

z <- table(train$Color1,train$Color3)
chisq.test(z)


## convert Type into Dog and Cat, gender into male, female, and mixed
## convert Color into text instead of numbers...

train$Type[train$Type ==  1] <- "Dog"
train$Type[train$Type ==  2] <- "Cat"

train$Gender[train$Gender ==  1] <- "Male"
train$Gender[train$Gender ==  2] <- "Female"
train$Gender[train$Gender ==  3] <- "Mixed"

## Breed2

sum(train$Breed2 != 0 )/sum(train$Breed2) 

## remove Name, PetID, RescuerID, and State becasue they are irrelevant
## and also remove Breed2, Color2, Color3
drop <- c("Name","PetID","RescuerID","Breed2","Color2", "Color3", "State")
DATA <- train[,!(names(train) %in% drop)]
DATA <- DATA[complete.cases(DATA), ]

## Breed
## Maine
Maine <- DATA[DATA$Breed1 == 276,]
rownames(Maine) <- NULL

ggplot(data = Maine, aes(fill = AdoptionSpeed, x = as.factor(Breed1), group =  AdoptionSpeed)) + 
  geom_bar()

## Persian
persian <- DATA[DATA$Breed1 == 285,]
rownames(persian) <- NULL
ggplot(data = persian, aes(fill = AdoptionSpeed, x = as.factor(Breed1), group =  AdoptionSpeed)) + 
  geom_bar()

## American Short Hair
ASH <- DATA[DATA$Breed1 == 243,]
rownames(ASH) <- NULL
ggplot(data = ASH, aes(fill = AdoptionSpeed, x = as.factor(Breed1), group =  AdoptionSpeed)) + 
  geom_bar()

## Poodle

Poodle <- DATA[DATA$Breed1 == 179,]
rownames(Poodle) <- NULL
ggplot(data = Poodle, aes(fill = AdoptionSpeed, x = as.factor(Breed1), group =  AdoptionSpeed)) + 
  geom_bar()

## Beagle
new <- DATA[DATA$Breed1 == 20,]
rownames(new) <- NULL
ggplot(data = new, aes(fill = AdoptionSpeed, x = as.factor(Breed1), group =  AdoptionSpeed)) + 
  geom_bar()

DATA %>%
  filter(Breed1 <= 30) %>%
  ggplot(aes(x = as.factor(Breed1))) + geom_bar() + theme(axis.text.x = element_text(angle=45,hjust=1,size=8))




## Since most records do not have breed2, so we decide to remove the column

## Dog vs. Cat

ggplot(data = DATA, aes(fill = AdoptionSpeed, x = Type, group =  AdoptionSpeed)) + 
  geom_bar()

numDog <- sum(DATA$Type == 1)
numCat <- sum(DATA$Type == 2)

## Vaccinated vs. AdoptionSpeed

ggplot(data = DATA, aes(fill = AdoptionSpeed, x = Vaccinated, group =  AdoptionSpeed)) + 
  geom_bar()
lm0 <- lm(AdoptionSpeed ~ Vaccinated, data = DATA)
summary(lm0)

########################################################################################
########################################################################################
###                        
###                        Business Understanding
###
## unsupervised analysis: clustering, plotting
## supervised analysis: predicting using different models,
## causal models: run predicting models on records of 3 or 4 regression speed, 
#                 select several highly significant variable, change related variables 
#                 to see the new performance. Through this we could learn which part and
#                 to what extent we could change to improve the animal's profile.

########################################################################################
###
###                        Data Prepration
###
########################################################################################
###
###                        Modeling
###
drop <- c("Description")
#DATA$Type <- as.factor(DATA$Type)
#DATA$Gender <- as.factor(DATA$Gender)
#DATA$Color1 <- as.factor(DATA$Color1)
#DATA$Color2 <- as.factor(DATA$Color2)
#DATA$Color3 <- as.factor(DATA$Color3)
#DATA$MaturitySize <- as.factor(DATA$MaturitySize)
#DATA$FurLength <- as.factor(DATA$FurLength)
#ATA$Vaccinated <- as.factor(DATA$Vaccinated)
#DATA$Dewormed <- as.factor(DATA$Dewormed)
#ATA$Sterilized <- as.factor(DATA$Sterilized)
#DATA$Health <- as.factor(DATA$Health)
#DATA$VideoAmt <- as.factor(DATA$VideoAmt)
#DATA$AdoptionSpeed <- as.factor(DATA$AdoptionSpeed)
#DATA$Quantity <- as.factor(DATA$Quantity)
#DATA$State <- as.factor(DATA$State)
#DATA$PhotoAmt <- as.factor(DATA$PhotoAmt)
#DATA$Breed1 <- as.factor(DATA$Breed1)
train_data <- DATA[,!(names(DATA) %in% drop)]

x_vars<- model.matrix(AdoptionSpeed~., data=train_data)[,-1]
##x_test<- model.matrix(~.,data = test_data)[,-1]
y_var <- train_data$AdoptionSpeed

###### Multinomial Logistic Regression with Lasso
cv_lasso <- cv.glmnet(x_vars,y_var,alpha = 1, family = 'multinomial')
cv_lasso
min_lambda <- cv_lasso$lambda.min
se1_lambda <- cv_lasso$lambda.1se
###### Multinomial Logistic Regression with Post Lasso
lasso <- glmnet(x_vars,y_var, family = 'multinomial')
lassobeta <- lasso$beta
features.min <- list()
lengthmin <- data.frame(betamin=rep(NA,length(lassobeta)))
features.1se <- list()
length1se <- data.frame(beta1se=rep(NA,length(lassobeta)))

##Extract Beta from Lasso results
i = 1
for (i in 1:length(lassobeta)) {
beta <- lassobeta[[i]]
features.min[[i]] <- support(beta[,which.min(cv_lasso$cvm)])
features.1se[[i]] <- support(beta[,which.min((cv_lasso$lambda-cv_lasso$lambda.1se)^2)])
lengthmin[i] <- length(features.min)
length1se[i] <- length(features.1se)
}

data.min <- list()
data.1se <- list()

i = 1
for (i in 1:length(lassobeta)) {
data.min[[i]] <- data.frame(x_vars[,features.min[[i]]],y_var)
data.1se[[i]] <- data.frame(x_vars[,features.1se[[i]]],y_var)
}

##### Cross Validation
set.seed(1)
nfold <- 10
n <- nrow(train_data)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

PL.OOS.ACC <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold)) 
L.OOS.ACC <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold)) 
ML.OOS.ACC <- data.frame(ML=rep(NA,nfold)) 
SVM.OOS.ACC <- data.frame(SVM=rep(NA,nfold)) 
RF.OOS.ACC <- data.frame(RF=rep(NA,nfold)) 

k = 1
for(k in 1:nfold){ 
  train <- which(foldid!=k) 
  
  ### Cross Validation for SVM
  svm_model <- svm(AdoptionSpeed~., data=train_data[train,], type="C-classification", kernel="linear")
  svm_pred <- predict(svm_model,newdata=train_data[-train,])
  svm_pred <- as.factor(svm_pred)
  svm.confusion <- confusionMatrix(svm_pred,as.factor(y_var[-train]))
  SVM.OOS.ACC$SVM[k] <- mean(data.frame(svm.confusion$byClass)$Balanced.Accuracy)
  
  ### Cross Validation for Random Forest
  #ran_for_model <- randomForest(AdoptionSpeed~.-Breed1, data=train_data[train,],importance = TRUE)
  #ran_for_pred <- predict(ran_for_model, train_data[-train,], type="class")
  #ran_for_pred <- as.factor(ran_for_pred)
  #ran_for_pred
  #ran_for.confusion <- confusionMatrix(ran_for_pred,as.factor(y_var[-train]))
  #RF.OOS.ACC$RF[k] <- mean(data.frame(ran_for.confusion$byClass)$Balanced.Accuracy)
  
  ### Cross Validation for Post Lasso Estimates
  
  ## Use selected variables to train data and make predictions using 5 models.
  rmin <- list()
  r1se <- list()
  predmin <- list()
  pred1se <- list()
  
  i = 1
  for (i in 1:length(lassobeta)) {
  rmin[[i]] <- multinom(y_var~., data=data.min[[i]], subset=train)
  r1se[[i]] <- multinom(y_var~., data=data.1se[[i]], subset=train)
  predmin[[i]] <- predict(rmin[[i]], newdata=data.min[[i]][-train,], type="probs")
  pred1se[[i]] <- predict(rmin[[i]], newdata=data.min[[i]][-train,], type="probs")
  }
  
  ## Calculate the average probability for each record in 5 models.
  summin <- data.frame(class1=rep(0,nrow(train_data[-train,])),
                    class2=rep(0,nrow(train_data[-train,])),
                    class3=rep(0,nrow(train_data[-train,])),
                    class4=rep(0,nrow(train_data[-train,])),
                    class5=rep(0,nrow(train_data[-train,])))
  sum1se <- data.frame(class1=rep(0,nrow(train_data[-train,])),
                       class2=rep(0,nrow(train_data[-train,])),
                       class3=rep(0,nrow(train_data[-train,])),
                       class4=rep(0,nrow(train_data[-train,])),
                       class5=rep(0,nrow(train_data[-train,])))
  avgprob.min <- data.frame(class1=rep(0,nrow(train_data[-train,])),
                    class2=rep(0,nrow(train_data[-train,])),
                    class3=rep(0,nrow(train_data[-train,])),
                    class4=rep(0,nrow(train_data[-train,])),
                    class5=rep(0,nrow(train_data[-train,])))
  avgprob.1se <- data.frame(class1=rep(0,nrow(train_data[-train,])),
                            class2=rep(0,nrow(train_data[-train,])),
                            class3=rep(0,nrow(train_data[-train,])),
                            class4=rep(0,nrow(train_data[-train,])),
                            class5=rep(0,nrow(train_data[-train,])))
  
  i = 1
  for (i in 1:length(lassobeta)){ ##iteratinon for 5 classes
    for (m in 1: length(lassobeta)){ ##iteration for 5 models
      summin[i] <- summin[i] + data.frame(predmin[m])[i]
      sum1se[i] <- sum1se[i] + data.frame(pred1se[m])[i]
    }
  }
  avgprob.min = summin / length(lassobeta)
  avgprob.1se = sum1se / length(lassobeta)
  
  postmin.result <- apply(avgprob.min,1,which.max) - 1
  post1se.result <- apply(avgprob.1se,1,which.max) - 1
  
  postlasso_min_pred <- as.factor(postmin.result)
  levels(postlasso_min_pred) <- c(levels(postlasso_min_pred),0)
  postlasso_1se_pred <- as.factor(post1se.result)
  levels(postlasso_1se_pred) <- c(levels(postlasso_1se_pred),0)
  postlasso_min.confusion <- confusionMatrix(postlasso_min_pred,as.factor(y_var[-train]))
  postlasso_1se.confusion <- confusionMatrix(postlasso_1se_pred,as.factor(y_var[-train]))
  PL.OOS.ACC$PL.min[k] <- mean(data.frame(postlasso_min.confusion$byClass)$Balanced.Accuracy)
  PL.OOS.ACC$PL.1se[k] <- mean(data.frame(postlasso_1se.confusion$byClass)$Balanced.Accuracy)
  
  ### Cross Validation for Lasso Estimates
  lasso_min <- glmnet(x_vars[train,],y_var[train], alpha = 1, lambda = min_lambda, family = 'multinomial')
  lasso_se1 <- glmnet(x_vars[train,],y_var[train], alpha = 1, lambda = se1_lambda, family = 'multinomial')
  predlassomin <- predict(lasso_min, newx=x_vars[-train,], type="class")
  predlasso1se  <- predict(lasso_se1, newx=x_vars[-train,], type="class")
  
  lasso_min_pred <- as.factor(predlassomin)
  lasso_1se_pred <- as.factor(predlasso1se)
  lasso_min.confusion <- confusionMatrix(lasso_min_pred,as.factor(y_var[-train]))
  lasso_1se.confusion <- confusionMatrix(lasso_1se_pred,as.factor(y_var[-train]))
  
  L.OOS.ACC$L.min[k] <- mean(data.frame(lasso_min.confusion$byClass)$Balanced.Accuracy)
  L.OOS.ACC$L.1se[k] <- mean(data.frame(lasso_1se.confusion$byClass)$Balanced.Accuracy)

  print(paste("Iteration",k,"of",nfold,"completed"))
}

ACCperformance <- cbind(PL.OOS.ACC,L.OOS.ACC,SVM.OOS.ACC)
m.OOS <- as.matrix(ACCperformance)
rownames(m.OOS) <- c(1:nfold)
barplot(colMeans(ACCperformance), las=3,xpd=FALSE, ylim=c(0, 0.65) , xlab="", ylab = bquote( "Accuracy performance"))


########################################################################################
###
###                        Deployment
###
########################################################################################
