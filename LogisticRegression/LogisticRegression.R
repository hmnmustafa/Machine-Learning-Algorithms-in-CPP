#Hamna Mustafa - hbm170002
#Sanika Buche - ssb170002
titanic <- read.csv("titanic_project.csv")

#turn into factors
titanic$survived <- as.factor(titanic$survived)

#no missing values
sapply(titanic, function(x) sum(is.na(x)==TRUE))

#split into train and test
train <- titanic[1:900,]
test <- titanic[901:1046,]

starttime <- proc.time()

#make model
glm1 <- glm(survived ~ pclass, data=train, family = "binomial")

endtime <- proc.time()

summary(glm1)

#evaluate on test data
probs <- predict(glm1, newdata = test, type = "response")

pred <- ifelse(probs>0.5, 1, 0)
acc <- mean(pred==test$survived)
print(paste("accuracy: ", acc))

library(caret)

confusionMatrix(as.factor(pred), reference = test$survived)

runtime <- endtime - starttime

runtime
