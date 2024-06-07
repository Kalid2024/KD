# 1. Install necessary packages
install.packages("tidyverse")
install.packages("readr")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("caret")
install.packages("randomForest")
install.packages("xgboost")
install.packages("pROC")
install.packages("sklearn.metrics")

# 1. Preprocessing and EDA
# A. Read the dataset
library(readr)
champs_data <- read_csv("C:\\Users\\Lbash\\Downloads\\champs.csv")

# B. Get the number of rows and columns
nrow(champs_data)
ncol(champs_data)

# C. Enumerate the columns
names(champs_data)

# D. Rename the columns
library(tidyverse)
champs_data <- champs_data %>%
  rename(case_type = dp_013,
         Underlying_Cause = dp_108,
         maternal_condition = dp_118)

# E. Rename the values
champs_data$case_type <- recode(champs_data$case_type,
                                "CH00716" = "Stillbirth",
                                "CH01404" = "Death in the first 24 hours",
                                "CH01405" = "Early Neonate (1 to 6 days)",
                                "CH01406" = "Late Neonate (7 to 27 days)",
                                "CH00718" = "Infant (28 days to less than 12 months)",
                                "CH00719" = "Child (12 months to less than 60 months)")

# F. Show the proportion of null values in each column
sapply(champs_data, function(x) sum(is.na(x)) / length(x))

# 2. Descriptive Data analysis
# A. Magnitude and proportion of each infant underlying cause for child death
table(champs_data$case_type)
prop.table(table(champs_data$case_type))

# B. Proportion and magnitude of the maternal factors contributing to child death
table(champs_data$maternal_condition)
prop.table(table(champs_data$maternal_condition))

# C. Proportion of the child death by the case type

ggplot(champs_data, aes(x = case_type)) +
  geom_bar(fill = "steelblue") +
  labs(x = "Case Type", y = "Count", title = "Proportion of Child Deaths by Case Type")

# 3. Correlation analysis
# Check the data types of the columns
str(champs_data)
# Convert the columns to numeric if possible
champs_data$Underlying_Cause <- as.numeric(champs_data$Underlying_Cause)
champs_data$maternal_condition <- as.numeric(champs_data$maternal_condition)

# Create the correlation matrix
corr_matrix <- cor(champs_data[, c("Underlying_Cause", "maternal_condition")])

# Create the correlation plot
corrplot(corr_matrix, method = "color", type = "upper")


# 4. Feature engineering
install.packages("caret")
install.packages("globals")
library(caret)
library(randomForest)
library(xgboost)

# A. Train the classification models
library(e1071)
model_svm <- svm(case_type ~ maternal_condition, data = champs_data)

# AdaBoost Classifier
library(sklearn)
model_ada <- AdaBoostClassifier(n_estimators = 100, random_state = 42)

# Random Forest Classifier
library(sklearn)
model_rf <- RandomForestClassifier(n_estimators = 100, random_state = 42)

# Gradient Boosting Classifier
library(sklearn)
model_gb <- GradientBoostingClassifier(n_estimators = 100, random_state = 42)

# XGBoost
library(xgboost)
model_xgb <- xgboost(data = as.matrix(champs_data[, c("maternal_condition")]),
                     label = champs_data$case_type,
                     nrounds = 100,
                     objective = "multi:softmax")
#####
model_lr <- glm(case_type ~ maternal_condition, data = champs_data, family = "multinomial")
model_svm <- svm(case_type ~ maternal_condition, data = champs_data)
model_ada <- AdaBoostClassifier(n_estimators = 100, random_state = 42)
model_rf <- RandomForestClassifier(n_estimators = 100, random_state = 42)
model_gb <- GradientBoostingClassifier(n_estimators = 100, random_state = 42)
model_xgb <- xgboost(data = as.matrix(champs_data[, c("maternal_condition")]),
                     label = champs_data$case_type,
                     nrounds = 100,
                     objective = "multi:softmax")

# B. Import the appropriate packages
library(sklearn)
library(xgboost)

# C. Rank the features based on their importance
importance(model_rf)
importance(model_xgb)

# 5. Model evaluation
library(caret)
library(pROC)

# A. Import the evaluation metric packages
library(sklearn.metrics)

# B. Perform n-fold cross-validation and select the best performing model
cv_results <- train(case_type ~ maternal_condition, data = champs_data, method = "rf", trControl = trainControl(method = "cv", number = 5))
best_model <- cv_results$finalModel

# C. Ensemble the models
ensemble_model <- stack([model_lr, model_svm, model_ada, model_rf, model_gb, model_xgb])

# D. Use Accuracy score to evaluate the performance
accuracy_score(champs_data$case_type, predict(best_model, champs_data))
accuracy_score(champs_data$case_type, predict(ensemble_model, champs_data))

# E. Plot the AUC and ROC curve
roc_curve(champs_data$case_type, predict(best_model, champs_data), plot = TRUE)
roc_curve(champs_data$case_type, predict(ensemble_model, champs_data), plot = TRUE)

# 6. Result Visualization
library(ggplot2)

# A. Plot the feature importance
feature_importance <- importance(model_rf)
ggplot(data.frame(feature = names(feature_importance), importance = feature_importance), aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Feature", y = "Importance", title = "Feature Importance")

# B. Plot the top five infant underlying causes of the child death
top_causes <- head(sort(table(champs_data$case_type), decreasing = TRUE), 5)
ggplot(data.frame(cause = names(top_causes), count = top_causes), aes(x = reorder(cause, count), y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Infant Underlying Cause", y = "Count", title = "Top 5 Infant Underlying Causes of Child Death")

# C. Plot the top five maternal factors contributing to the child death
top_maternal_factors <- head(sort(table(champs_data$maternal_condition), decreasing = TRUE), 5)
ggplot(data.frame(factor = names(top_maternal_factors), count = top_maternal_factors), aes(x = reorder(factor, count), y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Maternal Factor", y = "Count", title = "Top 5 Maternal Factors Contributing to Child Death")

# D. Plot the child death based on the case types
ggplot(champs_data, aes(x = case_type)) +
  geom_bar(fill = "steelblue") +
  labs(x = "Case Type", y = "Count", title = "Child Death by Case Type")