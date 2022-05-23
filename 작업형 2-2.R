library(dplyr)
library(randomForest)
library(e1071)
library(caret)

setwd("C:/Users/Samsung/Desktop/빅분기실기준비/220520")
main.ds <- read.csv(
    file = "TravelInsurancePrediction.csv",
    stringsAsFactor = TRUE,
    header = TRUE,
    fileEncoding = "UTF-8-BOM"
)

n <- nrow(main.ds)
train <- main.ds[1:1490, ]
test <- main.ds[1491:n, ]

nrow(test)

train$TravelInsurance <- as.factor(train$TravelInsurance)
head(train)
colSums(is.na(train))

set.seed(13579)
idx <- createDataPartition(
    train$TravelInsurance,
    p = 0.8,
    list = FALSE,
)

ds_train <- train[idx, ]
ds_valid <- train[-idx, ]

ds_train$TravelInsurance <- as.factor(ds_train$TravelInsurance)
ds_valid$TravelInsurance <- as.factor(ds_valid$TravelInsurance)


model_prePro <- preProcess(ds_train[, -10], method = c("range"))
scaled_ds_train <- predict(model_prePro, ds_train)
scaled_ds_valid <- predict(model_prePro, ds_valid)

md_svm <- svm(
    TravelInsurance ~., 
    data = scaled_ds_train, 
    probability = TRUE
)

pred_svm <- predict(
    object = md_svm,
    newdata = scaled_ds_valid, 
    probability = TRUE
)

caret::confusionMatrix(
    data = pred_svm,
    refer = scaled_ds_valid$TravelInsurance
)

md_rf <- randomForest(
    TravelInsurance ~ .,
    data = scaled_ds_train,
    ntree = 300,
    do.trace = T,
    probability = TRUE
)

pred_rf <- predict(
    md_rf,
    newdata = scaled_ds_valid,
    probability = TRUE,
    type = "response"
)


set.seed(13579)
md_fit <- randomForest::randomForest(
    TravelInsurance ~.,
    data = train,
    ntree = 300,
    do.trace = T,
    probability = TRUE
)

# 마지막 결과 예측의 행 수는 test의 nrow와 같아야 함

pred_fit <- predict(
    md_fit,
    newdata = test,
    probability = TRUE,
    type = "prob"
)

head(pred_fit)
nrow(test)
nrow(pred_fit)

result <- data.frame(
    c(1:nrow(test)),
    pred_fit[, 2])

nrow(result)

result
colnames(result) <- c("index", "y_pred")
head(result)

write.csv(result, "0000.csv", row.names = FALSE)
result <- read.csv("0000.csv")
result <- head(result, 10)
print(result)