#Iphone Cell Sentiment
install.packages("caret")
library(caret)
library(RGtk2)
install.packages("rattle")
library(rattle)

#Cell Data Install
Cell_Iphone <- read.csv("/Users/timwilson/Documents/Cellsent/emr output/iPhoneLargeMatrix.csv.csv")
setwd("~/Desktop/Austin DA/Course 3/Task 3/")

#Data Exploration 
View(Cell_Iphone)
names(Cell_Iphone)
summary(Cell_Iphone)
str(Cell_Iphone)
is.na(Cell_Iphone)

#Data Visualization
install.packages("corrplot")
library(corrplot)
Iphone_corr <- cor(Cell_Iphone)
Iphone_corr
corrplot(cor(Cell_Iphone), order ="hclust")

#Feature Selection (Optional)
#Method One - FitControl
library(caret)
?trainControl
fitcontrol_iphone <- trainControl(method = "oob", number = 10)
?train
rf_Iphone <- train(Cell_Iphone, Cell_Iphone$iphoneSentiment, method="rf", trControl=fitcontrol_iphone)
predictors(rf_Iphone)


#Method 2- Correlation 
library(caret)
descrCor_Iphone <- cor(Cell_Iphone)
highcorr_iphone <- sum(descrCor_Iphone[upper.tri(descrCor_Iphone)])
summary(highcorr_iphone)
highlyCorDescr_iphone <- findCorrelation(descrCor_Iphone, cutoff = .80)
filteredDescr_iphone <- Cell_Iphone[,-highlyCorDescr_iphone]
summary(filteredDescr_iphone)
names(filteredDescr_iphone)
Iphone_final <- filteredDescr_iphone
names(Iphone_final)

#Preprocess Matrices - Discretization
install.packages("arules")
install.packages("arulesViz")
update.packages("arulesViz")

library(arules)
library(arulesViz)
library(devtools)

disfixed7_iphone <- discretize(Iphone_final$iphoneSentiment, "fixed", categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf), labels = c("very negative", "somewhat negative", "negative", "neutral", "somewhat positive", "positive", "very positive"))
summary(disfixed7_iphone)
str(disfixed7_iphone)
Iphone_final$iphoneSentiment <-disfixed7_iphone
summary(Iphone_final$iphoneSentiment)
Iphone_Tim <- Iphone_final
summary(Iphone_Tim)
names(Iphone_Tim)
Iphone_final <- Iphone_Tim

#Collect Sentiment Counts 
summary(Iphone_final$iphoneSentiment)
summary(Iphone_final)
names(Iphone_final)
plot(Iphone_final$iphonecamneg)

#Plot Data and Data Visualization - ggplot
install.packages("ggplot2")
library(ggplot2)
library(caret)
install.packages("grDevices")
library(grDevices)
Iphone_plot<- plot.default(Iphone_final, Iphone_final$iphoneSentiment, method = "graph",)
plot(Iphone_final$iphoneSentiment)
iphone_bar <- autoplot(Iphone_final$iphoneSentiment)
hist(Iphone_final$iphoneSentiment, Iphone_final)
heatmap(Iphone_final$iphoneSentiment)

#Model Development
#10 fold cross validation
set.seed(123)
library(caret)
fitcontrol_iphone <- trainControl(method = "oob", number = 10)
fitcontrol_iphone

#Data Partition 
#Trained Model 70/30 split 

set.seed(123)
inTrain_iphone <- createDataPartition(Iphone_final$iphoneSentiment, p = .70, list = FALSE)
inTrain_iphone
training_type_iphone <- Iphone_final[inTrain_iphone,]
testing_type_iphone <- Iphone_final[-inTrain_iphone,]
training_type_iphone
testing_type_iphone

#Sample for data partition 
train_ind_iphone <- sample(seq_len(nrow(inTrain_iphone)), size = 4000)

training_type_iphone_sample <-Iphone_final[train_ind_iphone,]
testing_type_iphone_sample <- Iphone_final[-train_ind_iphone,]


#C5.0
install.packages("C50")
library(C50)
set.seed(123)
c5model_iphone<- C5.0.default(x= training_type_iphone, y = training_type_iphone$iphoneSentiment, trials = 1, rules = FALSE, weights = NULL,
                       control = C5.0Control(), costs = NULL)
summary(c5model_iphone)
C5pred_iphone <- predict(c5model_iphone, testing_type_iphone)
C5pred_iphone
summary(C5pred_iphone)
postResample(C5pred_iphone, testing_type_iphone$iphoneSentiment)


#Random Forrest
set.seed(123)
rf_Iphone <- train(Iphone_final, Iphone_final$iphoneSentiment, method="rf", trControl=fitcontrol_iphone)
rf_Iphone
rfpred_iphone <- predict(rf_Iphone, testing_type_iphone)
summary(rfpred_iphone)
postResample(rfpred_iphone, testing_type_iphone$iphoneSentiment)

#KNN
library(caret)
set.seed(123)
fitControl_knn <- trainControl(method = "repeatedcv", number = 10)
knnFit_cell_iphone <- train(iphoneSentiment~., data = training_type_iphone, method = "knn", trControl=fitControl_knn, preProcess = c("center","scale"), tuneLength = 20)
knnFit_cell_iphone
knnpred_iphone <- predict(knnFit_cell_iphone, testing_type_iphone_sample)
knnpred_iphone
postResample(knnpred_iphone, testing_type_iphone_sample$iphoneSentiment)

#SVM
set.seed(123)
SVMFit_cell_iphone <- train(iphoneSentiment~., data = training_type_iphone, method = "svmLinear", trControl=fitControl_knn, preProcess =c("center", "scale"), tuneLength = 10)
SVMFit_cell_iphone
SVMpred_iphone <- predict(SVMFit_cell_iphone, testing_type_iphone)
SVMpred_iphone
summary(SVMpred_iphone_opt)
plot(SVMpred_iphone)
postResample(SVMpred_iphone, testing_type_iphone$iphoneSentiment)

#Optimized Model - Iphone - SVM Model
set.seed(123)
SVMFit_optimized <- train(iphoneSentiment~., data = training_type_iphone, method = "svmLinear", trControl=fitControl_knn, preProcess =c("center", "scale"), tuneLength = 10)
SVMFit_optimized
summary(SVMFit_optimized)
SVMpred_iphone_opt <- predict(SVMFit_optimized, testing_type_iphone)
SVMpred_iphone_opt
summary(SVMpred_iphone_opt)

postResample(SVMpred_iphone_opt, testing_type_iphone$iphoneSentiment)

install.packages("ggplot2")
library(ggplot2)
library(caret)
install.packages("grDevices")
library(grDevices)

plot(SVMpred_iphone_opt)
