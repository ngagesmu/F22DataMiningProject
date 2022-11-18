##########################################
#                                        #
# Fall 2022 Data Mining -- Group Project #
#                                        #
# Team:                                  #
#   Tania Cuff                           #
#   Nathan Gage                          #
#   Timothy Lee                          #
#   Elizabeth McPherson                  #
#                                        #
##########################################

# Exploratory Data Analysis

columnnames <- c(
  "age",
  "workclass",
  "fnlwgt",
  "education",
  "education-num",
  "marital-status",
  "occupation",
  "relationship",
  "race",
  "sex",
  "capital-gain",
  "capital-loss",
  "hours-per-week",
  "native-country",
  "income"
)

# reading the data
train <-
  read.csv("adult.data",
           header = FALSE,
           na.strings = c("?", " ?", "NA"),)
test <-
  read.csv(
    "adult.test",
    header = FALSE,
    skip = 1,
    na.strings = c("?", " ?", "NA"),
  )

# naming columns
names(train) <- columnnames
names(test) <- columnnames

# converting char columns to factors
train <- as.data.frame(unclass(train), stringsAsFactors = T)
test <- as.data.frame(unclass(test), stringsAsFactors = T)

# needed because test is missing one country to create factors from
allcountries <-
  unique(union(train$native.country, test$native.country))
test$native.country <-
  factor(test$native.country, levels = allcountries)

str(train)
str(test)

# VERIFY DATA QUALITY

# missing values / invalid:
colSums(is.na(train))
train <- na.omit(train)
colSums(is.na(train))

colSums(is.na(test))
test <- na.omit(test)
colSums(is.na(test))

# checking for outliers:

# no outliers in age
summary(train$age)
summary(test$age)

# seems like train & test have some rows where capital.gain == 99999
# this is a suspiciously high & specific value so we will omit it
summary(train$capital.gain)
summary(test$capital.gain)

nrow(train[train$capital.gain == 99999, ])
nrow(test[test$capital.gain == 99999, ])

train <- train[!(train$capital.gain == 99999), ]
test <- test[!(test$capital.gain == 99999), ]

# the values are now far more reasonable
summary(train$capital.gain)
summary(test$capital.gain)

# no outliers in cap loss
summary(train$capital.loss)
summary(test$capital.loss)

# all values are > 0 and < 7 * 24 = 168
summary(train$hours.per.week)
summary(test$hours.per.week)

# drop education.num (same as our education factor)
train <- train[, !names(train) %in% c("education.num")]
test <- test[, !names(test) %in% c("education.num")]

# GIVE SIMPLE AND APPROPRIATE STATISTICS

# we already performed some of this in the data clean up,
# however, we will take a look at the finalized data now:
summary(train)

# VISUALIZE SOME OF THE FEATURES
library(ggplot2)

ggplot(train, aes(x = factor(income))) +
  geom_bar(color = "black", fill = "white") +
  labs(title = "counts of income type",
       subtitle = "across training set",
       x = "income level",
  )

# distribution of age
ggplot(train, aes(x = age)) +
  geom_histogram(aes(y = ..density..), color = "black", fill = "white") +
  geom_density(alpha = .2) +
  labs(title = "distribution of ages",
       subtitle = "across training set",
       x = "age",
  )

# let's take a look at the different types of jobs:
ggplot(train, aes(x = factor(occupation))) +
  geom_bar(color = "black", fill = "white") +
  labs(title = "counts of occupation types",
       subtitle = "across training set",
       x = "occupation") +
  theme(axis.text.x = element_text(
    angle = 90,
    vjust = 0.5,
    hjust = 1
  ))


# EXPLORE RELATIONSHIPS

# first, we're curious how age affects the income status
ggplot(train, aes(x = age)) +
  geom_histogram(aes(y = ..density..), color = "black", fill = "white") +
  geom_density(alpha = .2) +
  facet_grid(~ income) +
  labs(title = "distribution of ages by income level",
       subtitle = "across training set",
       x = "age",
  )

ggplot(train, aes(
  x = age,
  group = factor(income),
  fill = factor(income)
)) +
  geom_histogram(aes(y = ..density..), position = "dodge", color = "black") +
  geom_density(alpha = .2) +
  labs(
    title = "distribution of ages by income level",
    subtitle = "across training set",
    x = "age",
    fill = "income level"
  )

# for our own sake, does education level affect income?
ggplot(train, aes(
  x = factor(
    education,
    levels = c(
      " Preschool",
      " 1st-4th",
      " 5th-6th",
      " 7th-8th",
      " 9th",
      " 10th",
      " 11th",
      " 12th",
      " HS-grad",
      " Assoc-voc",
      " Assoc-acdm",
      " Some-college",
      " Bachelors",
      " Masters",
      " Prof-school",
      " Doctorate"
    )
  ),
  group = factor(income),
  fill = factor(income),
)) +
  geom_bar(position = "dodge",
           color = "black") +
  labs(
    title = "counts of education levels by income level",
    subtitle = "across training set",
    x = "education",
    fill = "income level"
  ) +
  theme(axis.text.x = element_text(
    angle = 90,
    vjust = 0.5,
    hjust = 1
  ))

# and finally we are interested in seeing the relationship between hours per week and income
ggplot(train, aes(x = factor(income), fill = factor(income))) +
  geom_bar(aes(y = (..count.. / sum(..count..))), color = "black") +
  facet_grid(~ sex) +
  labs(
    title = "income levels by gender",
    subtitle = "across training set",
    x = "income level",
    y = "density",
    fill = "income level"
  )

# CLASSIFICATION

# model 1: LOGISTIC REGRESSION
logmodel <- glm(income ~ ., data = train, family = "binomial")
logmodel.preds <- predict(logmodel, test, type = "response")
logmodel.confusion <-
  table(test$income, ifelse(logmodel.preds < 0.5, 0, 1))
logmodel.accuracy <-
  (logmodel.confusion[1] + logmodel.confusion[4]) / sum(logmodel.confusion)
logmodel.confusion
logmodel.accuracy

# model 2: RANDOM FOREST
library(randomForest)
rfmodel <- randomForest(income ~ ., data = train)
rfmodel.preds <- predict(rfmodel, test)
rfmodel.confusion <- table(test$income, rfmodel.preds)
rfmodel.accuracy <-
  (rfmodel.confusion[1] + rfmodel.confusion[4]) / sum(rfmodel.confusion)
rfmodel.confusion
rfmodel.accuracy

# model 3: SVM
library('e1071')
svmmodel <-
  svm(
    income ~ .,
    data = train,
    type = "C-classification",
    kernel = "linear",
    scale = T
  )
svmmodel.preds <- predict(svmmodel, test)
svmmodel.confusion <- table(test$income, svmmodel.preds)
svmmodel.accuracy <-
  (svmmodel.confusion[1] + svmmodel.confusion[4]) / sum(svmmodel.confusion)
svmmodel.confusion
svmmodel.accuracy

svmmodel2 <-
  svm(
    income ~ .,
    data = train,
    type = "C-classification",
    kernel = "polynomial",
    scale = T
  )
svmmodel2.preds <- predict(svmmodel2, test)
svmmodel2.confusion <- table(test$income, svmmodel2.preds)
svmmodel2.accuracy <-
  (svmmodel2.confusion[1] + svmmodel2.confusion[4]) / sum(svmmodel2.confusion)
svmmodel2.confusion
svmmodel2.accuracy

svmmodel3 <-
  svm(
    income ~ .,
    data = train,
    type = "C-classification",
    kernel = "sigmoid",
    scale = T
  )
svmmodel3.preds <- predict(svmmodel3, test)
svmmodel3.confusion <- table(test$income, svmmodel3.preds)
svmmodel3.accuracy <-
  (svmmodel3.confusion[1] + svmmodel3.confusion[4]) / sum(svmmodel3.confusion)
svmmodel3.confusion
svmmodel3.accuracy

# final model is logistic regression model with 84.67855% accuracy
# honorable mentions: SVM w/ linear kernel & SVM with sigmoid kernel

# creative work: ROC analysis
library(ROCR)
pred <- prediction(logmodel.preds, test$income)
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize = T, lwd = 2)
abline(a = 0, b = 1)

# this is a good ROC, with a solid balance between FPR and TPR. 
# given that we have imbalanced classes, this means we are acheiving
# good performance despite class imbalance.