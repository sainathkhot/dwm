# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Load the Wine dataset
wine = load_wine()
X = wine.data  # Features
y = wine.target  # Labels

# Step 2: Display the dataset in a DataFrame
wine_df = pd.DataFrame(X, columns=wine.feature_names)
wine_df['target'] = y
print("Wine Dataset :")
print(wine_df.head())  # Show first 5 rows of the dataset

# Step 3: Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a Decision Tree classifier model
dt_model = DecisionTreeClassifier(random_state=42)

# Step 5: Train the model using the training data
dt_model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = dt_model.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(15,3))
tree.plot_tree(dt_model, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.show()
