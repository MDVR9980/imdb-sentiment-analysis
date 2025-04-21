import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

# Load the preprocessed data
df = pd.read_csv("cleaned_IMDB_Dataset.csv")  # Make sure this file exists

# Step 1: Split dataset into training and testing sets
X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Convert labels to binary

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 4: Make predictions and evaluate
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Step 6: Plot precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
labels = ['Negative', 'Positive']
x = range(len(labels))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar([i - width for i in x], precision, width=width, label='Precision')
plt.bar(x, recall, width=width, label='Recall')
plt.bar([i + width for i in x], f1, width=width, label='F1-Score')
plt.xticks(x, labels)
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('Evaluation Metrics per Class')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Step 7: Show misclassified examples
X_test_df = X_test.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)

results = pd.DataFrame({
    'review': X_test_df,
    'true_label': y_test_df,
    'predicted_label': y_pred
})

incorrect_predictions = results[results['true_label'] != results['predicted_label']]
print("\nSome misclassified reviews:\n")
print(incorrect_predictions.sample(5, random_state=42))