#load packages
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)


# load dataset
dataset <- read.csv("C:/Users/Hp/Downloads/data 1/data/bank-full.csv", sep = ";")
print(dataset)

# Exploratory data analysis

# preview dataset
head(dataset, 10)
str(dataset)

# Insight:
# There are a total of 45211 rows with 17 columns


# find total no. of rows and columns
dim(dataset)
cat('Number of rows:', nrow(dataset), '\n')
cat('Number of columns:', ncol(dataset))


# datatype for each column
datatypes <- sapply(dataset, class)
datatypes

# dealing with missing values and  empty values
missing_values <- colSums(is.na(dataset))
print(missing_values)

empty_values <- colSums(dataset == "")
empty_values

sum(dataset == "")

# INSIGHT: There are no missing or empty values

# dealing with duplicates
dup <- sum(duplicated(dataset))
print (dup)                              # NO duplicates found


# summary
dataset_summary <- summary(dataset)
print(dataset_summary)


# select numeric columns
numeric_col <- names(dataset)[sapply(dataset, is.numeric)]
numeric_col

# boxplots for outlier visualization
par(mfrow = c(3, 3))
for (col in numeric_col) {
  boxplot(dataset[[col]], main = paste("Boxplot of", col), col = "blue")
}

#Insight:
# With the numerical columns, age, balance, duration, campaign, pdays and previous
# show some outliers.



# Histogram for numeric features
for (num_col in numeric_col) {
  p <- ggplot(dataset, aes_string(x = num_col, fill = "y")) +
    geom_histogram(position = "dodge", stat = "bin", bins = 20, color = "black") + # Use 'geom_histogram'
    theme_minimal() +
    ggtitle(paste("Relationship between", num_col, "and y")) +
    xlab(num_col) +
    ylab("Frequency") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for readability
  print(p)
}


# select categorical columns
categorical_col <- names(dataset)[sapply(dataset, is.character)]
categorical_col

# Bar plots for categorical features
for (cat_col in categorical_col) {
  p <- ggplot(dataset, aes_string(x = cat_col, fill = "y")) +
    geom_bar(position = "dodge") +
    theme_minimal() +
    ggtitle(paste("Relationship between", cat_col, "and y")) +
    xlab(cat_col) +
    ylab("Count")
  print(p)
}


# Insight:
# Majority of clients did not subscript to the Bank's term deposit.
# However, campaign and contact shows a bit improvement. This could be
# due to consistent reaching out through cellular.


# checking the number of yes and no
y_count <- count(dataset, y)
print(y_count)

# Insight:
# This confirm the previous insight, most client did not subscribe.
# This makes the data imbalance.



# FEATURE ENGINEERING

# Apply label encoding to the categorical column
for (col in categorical_col) {
  dataset[[col]] <- as.integer(factor(dataset[[col]]))
}

# Check the result
head(dataset)

# Compute the correlation matrix
cor_matrix <- cor(dataset)
print(cor_matrix)


# Convert correlation matrix to dataframe
cor_matrix_df <- as.data.frame(as.table(cor_matrix))

names(cor_matrix_df) <- c("Var1", "Var2", "Correlation")

# Visualize the correlation matrix as a heatmap
ggplot(data = cor_matrix_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "lightpink", high = "hotpink") +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        axis.text.y = element_text(angle = 0, hjust = 1))

# Insight:
# y is positively correlated with duration of call and also
# shows slight positive correlation with previous and pdays.
# This means clients subscribe to term deposits after gaining sufficient understanding



# Separate the target (y) variable from the rest
X <- dataset[, !colnames(dataset) %in% "y"]
y <- dataset$y

# Split the dataset into training and testing sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)  # 80% for training

X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Display the shapes of the resulting splits
cat("X_train shape:", dim(X_train), "\n")
cat("X_test shape:", dim(X_test), "\n")
cat("y_train shape:", length(y_train), "\n")
cat("y_test shape:", length(y_test), "\n")



# TRAINING A DECISION TREE MODEL

# Train a Decision Tree model
model <- rpart(y_train ~ ., data = cbind(X_train, y_train), method = "class", parms = list(split = "gini"))

# Display the trained Decision Tree model
print(model)


# PREDICTION AND EVALUATION

# Make predictions on the test set
predictions <- predict(model, X_test, type = "class")

# Generate the confusion matrix and classification report
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(y_test))

# Print the confusion matrix
cat("Confusion Matrix:\n")
print(conf_matrix$table)

# Print classification metrics
cat("\nClassification Report:\n")
print(conf_matrix$byClass)
print(conf_matrix$overall)


# Insight:
# The model has a high accuracy (88.53%) but a low specificity (17.94%)
# This means it is biased toward predicting the positive class (yes).
# The sensitivity is very high (98%), which is good, but it's inability
# to correctly identify the negative class (low specificity) limits its overall performance.
# Also the Kappa value of 0.22 and the balanced accuracy of 0.58 suggest
# that the model might not be performing optimally.
# Precision and F1-score are good, suggesting the model is good at predicting
# the positive class.


# To avoid bias in prediction, we will perform re-sampling using undersampling
# techniques. This will match the size of both class to be equal.
# That is, the majority class (no) will be reduce to match the size of
# the minority class (yes).



# UNDERSAMPLING METHOD

# Combine the dataset
data <- cbind(X_train, y_train)

# Check the class distribution
print(table(data$y_train))

# Separate minority and majority class
minority_class <- data[data$y_train == 2, ]
majority_class <- data[data$y_train == 1, ]


# Undersample the majority class (reduce it to match the minority class)
undersampled_data <- rbind(
  minority_class,
  majority_class[sample(1:nrow(majority_class), size = nrow(minority_class), replace = FALSE), ]
)

# Check the new class distribution
print(table(undersampled_data$y_train))

# Split back into features and target
X_balanced <- undersampled_data[, -ncol(undersampled_data)]
y_balanced <- undersampled_data[, ncol(undersampled_data)]

# Train a decision tree model
decision_tree <- rpart(y_balanced ~ ., data = data.frame(X_balanced, y_balanced), method = "class")
print(decision_tree)
# Ensure test data has the same feature names
colnames(X_test) <- colnames(X_balanced)

# Predict on test data
predictions <- predict(decision_tree, newdata = X_test, type = "class")

# Evaluate using a confusion matrix
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(y_test))

# Print the classification report
cat("\nClassification Report:\n")
print(conf_matrix$byClass)
print(conf_matrix$overall)


# Insight:
# After Undersampling, the model is more balanced, with improved specificity,
# precision, and balanced accuracy. This implies that the model became fair
# across both classes.

