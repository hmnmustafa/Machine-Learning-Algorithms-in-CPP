titanic <- read.csv("C:\\Users\\sanik\\Desktop\\CodingFiles\\mlhomework\\homework5\\titanic_project.csv")

#turn into factors
attach(titanic)
titanic$survived <- as.factor(titanic$survived)
titanic$pclass <- as.factor(titanic$pclass)
titanic$sex <- as.factor(titanic$sex)

#no missing values
sapply(titanic, function(x) sum(is.na(x)==TRUE))

#split into train and test
train <- titanic[1:900,]
test <- titanic[901:1046,]

library(e1071)

starttime <- proc.time()

nb1 <- naiveBayes(survived ~ pclass + sex + age, data = train)
nb1

p1 <- predict(nb1, newdata = test, type = "class")

endtime <- proc.time()

library(caret)

confusionMatrix(p1, reference = test$survived)

runtime <- endtime - starttime

print(paste("Runtime of Naive Bayes: ", runtime[1]))