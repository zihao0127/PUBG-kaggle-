# PUBG-kaggle-
kaggle 2/4
library(dplyr)
library(caret)
library(xgboost)
library(data.table)
library(readr)
library(ggplot2)
library(tidyverse)
#### read csv ####
setwd("C:/Users/zihao/OneDrive/Desktop/PUBG kaggle")
training <- fread("train_v2.csv")
training <- training[complete.cases(training),]
### look through data ###
table(training$matchType)
#### before pred ####
training <- training %>%
  mutate(winPlacePerc = ifelse(is.na(winPlacePerc), 0, winPlacePerc)) #replace missing#
#seperate match type as solo, duo , squad,flare and crash cbind fpp and tpp #
training$matchType <- gsub(".*solo.*", "solo", training$matchType)
training$matchType <- gsub(".*duo.*", "duo", training$matchType)
training$matchType <- gsub(".*squad.*", "squad", training$matchType)
training$matchType <- gsub(".*crash.*", "other", training$matchType)
training$matchType <- gsub(".*flare.*", "other", training$matchType)
#fix some importance col and add some col
training$energy <- training$heals + training$boosts # totale boosting 
training$movedistance <- training$walkDistance + training$rideDistance + training$swimDistance
training$rankPoints <- ifelse(training$rankPoints == -1,0,training$rankPoints) #remove those -1
training$realplayer <- ifelse(training$movedistance == 0,0,1) #did they really played this game
training$shotacc <- ifelse(training$kills == 0, 0, training$headshotKills / training$kills) #headshotkills can show the player's accuracy


##### xgb model #####
####solo model####
solo <- training[training$matchType == "solo",]
solo <- solo %>%
  group_by(matchId)%>%
  mutate(maxKP = killPlace/maxPlace)
set.seed(77)
solo_inTrain <- createDataPartition(y = solo$winPlacePerc,p = 0.8,list = F)
solo_train <- solo[solo_inTrain,]
solo_test <- solo[-solo_inTrain,]  # set train and test set

summary(solo_train)
rm(solo_inTrain)
#keep some variables#
solo_train <- subset(solo_train, select = -c(Id, groupId, matchId, assists, DBNOs, matchType, numGroups, 
                                             roadKills, swimDistance, teamKills, vehicleDestroys))
solo_test <- subset(solo_test, select = -c(Id, groupId, matchId, assists, DBNOs, matchType, numGroups, 
                                           roadKills, swimDistance, teamKills, vehicleDestroys))
#xgb_solo#
solo_train2 <- subset(solo_train, select = -c(winPlacePerc))
train_solo_xgb.label <- solo_train$winPlacePerc 
solo_train.matrix <- xgb.DMatrix(data = as.matrix(solo_train2), 
                                 label = train_solo_xgb.label)
solo_test2 <- subset(solo_test, select = -c(winPlacePerc))
test_solo_xgb.label <- solo_test$winPlacePerc
solo_test.matrix <- xgb.DMatrix(data = as.matrix(solo_test2, 
                                label = test_solo_xgb.label))
#set train and test matrix
xgb.solo_model <- xgboost(data = solo_train.matrix, nrounds = 800, 
                     booster = "gbtree", objective = "reg:linear")
xgb.solo_pred <- predict(xgb.solo_model, solo_test.matrix)
mean(abs(xgb.solo_pred - solo_test$winPlacePerc)) #0.04121475 solo only#


rm(solo_test,solo_test2,solo_train,solo_train2,xgb.solo_pred,train)

#### team model ####
team <- training[training$matchType == "duo"|training$matchType == "squad"|training$matchType == "other",] #load df
team <- team %>%
    group_by(groupId)%>%
    mutate(maxkill = max(killPlace),
    meankill = mean(killPlace), 
    maxdist = max(movedistance), 
    meandist = mean(movedistance),
    meanvehcile = mean(rideDistance),
    meanacc = mean(shotacc),
    totalenergy = sum(energy), 
    totaldamage = sum(damageDealt), 
    totalkill = sum(kills),
    totalwepon = sum(weaponsAcquired)) # add importance variables
#creat train and validation set
set.seed(77)
team_inTrain <- createDataPartition(y = team$winPlacePerc,p = 0.8,list = F)
team_train <- team[team_inTrain,]
team_test <- team[-team_inTrain,]  # set train and test set
rm(team_inTrain)
#keep and deleat some variables#
team_train <- subset(team_train, select = -c(Id, groupId, matchId,matchType))
team_test <- subset(team_test, select = -c(Id, groupId, matchId,matchType))

#xgb_team#
#train
team_train2 <- subset(team_train, select = -c(winPlacePerc))
train_team_xgb.label <- team_train$winPlacePerc 
team_train.matrix <- xgb.DMatrix(data = as.matrix(team_train2), 
                                 label = train_team_xgb.label)
#test
team_test2 <- subset(team_test, select = -c(winPlacePerc))
test_team_xgb.label <- team_test$winPlacePerc
team_test.matrix <- xgb.DMatrix(data = as.matrix(team_test2, 
                                                 label = test_team_xgb.label))
#set train and test matrix
xgb.team_model <- xgboost(data = team_train.matrix, nrounds = 1000, 
                          booster = "gbtree", objective = "reg:linear")
xgb.team_pred <- predict(xgb.team_model, team_test.matrix)
mean(abs(xgb.team_pred - team_test$winPlacePerc)) #0.04457687#








#### write kaggle submission ####
#load test set
testing <- fread("test_v2.csv")
testing <- testing[complete.cases(testing),]
#seperate match type#
testing$matchType <- gsub(".*solo.*", "solo", testing$matchType)
testing$matchType <- gsub(".*duo.*", "duo", testing$matchType)
testing$matchType <- gsub(".*squad.*", "squad", testing$matchType)
testing$matchType <- gsub(".*crash.*", "other", testing$matchType)
testing$matchType <- gsub(".*flare.*", "other", testing$matchType)
#fix some importance col and add some col
testing$energy <- testing$heals + testing$boosts # totale boosting 
testing$movedistance <- testing$walkDistance + testing$rideDistance + testing$swimDistance
testing$rankPoints <- ifelse(testing$rankPoints == -1,0,testing$rankPoints) #remove those -1
testing$realplayer <- ifelse(testing$movedistance == 0,0,1) #did they really played this game
testing$shotacc <- ifelse(testing$kills == 0, 0, testing$headshotKills / testing$kills) #headshotkills can show the player's accuracy

#solo predict
solo_kaggle.sb <- testing[testing$matchType == "solo",]
solo_kaggle <- solo_kaggle.sb %>%
  group_by(matchId)%>%
  mutate(maxKP = killPlace/maxPlace)
solo_kaggle <- subset(solo_kaggle, select = -c(Id, groupId, matchId, assists, DBNOs, matchType, numGroups, 
                                           roadKills, swimDistance, teamKills, vehicleDestroys))

solo_kaggle$winPlacePerc <- predict(xgb.solo_model, data.matrix(solo_kaggle)) #predict solo
solo_kaggle$winPlacePerc <- ifelse(solo_kaggle$winPlacePerc >1, 1, solo_kaggle$winPlacePerc)
solo_kaggle$winPlacePerc <- ifelse(solo_kaggle$winPlacePerc <0, 0, solo_kaggle$winPlacePerc)

#team predict
team_kaggle.sb <- testing[testing$matchType == "duo"|testing$matchType == "squad"|testing$matchType == "other",] #load df
team_kaggle <- team_kaggle.sb %>%
  group_by(groupId)%>%
  mutate(maxkill = max(killPlace),
         meankill = mean(killPlace), 
         maxdist = max(movedistance), 
         meandist = mean(movedistance),
         meanvehcile = mean(rideDistance),
         meanacc = mean(shotacc),
         totalenergy = sum(energy), 
         totaldamage = sum(damageDealt), 
         totalkill = sum(kills),
         totalwepon = sum(weaponsAcquired)) 
team_kaggle <- subset(team_kaggle, select = -c(Id, groupId, matchId,matchType))
team_kaggle$winPlacePerc <- predict(xgb.team_model, data.matrix(team_kaggle))
team_kaggle$winPlacePerc <- ifelse(team_kaggle$winPlacePerc > 1, 1, team_kaggle$winPlacePerc)
team_kaggle$winPlacePerc <- ifelse(team_kaggle$winPlacePerc < 0, 0, team_kaggle$winPlacePerc)

#### submission ####
solo_submit <- cbind(solo_kaggle.sb$Id, solo_kaggle$winPlacePerc)
team_submit <- cbind(team_kaggle.sb$Id, team_kaggle$winPlacePerc)
submission <- rbind(solo_submit, team_submit)
submission <- as.data.frame(submission)
colnames(submission) <- c("Id", "winPlacePerc")
write.csv(submission, "C:/Users/zihao/OneDrive/Desktop/PUBG kaggle/kaggle_submission.csv", row.names = F)



####5-fold CV ####
train_control <- trainControl(method="cv", number=5,
                              savePredictions = TRUE,verboseIter = T)
solo_model_fit <- train(winPlacePerc~., data = solo_train,
                        trControl = train_control, method = "lm")

solo_lm_predict <- predict(solo_model_fit, newdata = solo_test)
mean(abs(solo_lm_predict - solo_test$winPlacePerc))
