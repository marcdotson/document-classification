#import neccessary libraies
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score


#-----------------------------------------------------------------------------------------------
# Model Evaluation
#-----------------------------------------------------------------------------------------------

# Testing the best estimator on the test set based off of prior grid search or any best performing model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Get predicted probabilities for the positive class (1)
y_prob = best_model.predict_proba(X_test)[:, 1]  # Use [:, 1] for the positive class

#print our classification report 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC AUC score and accuracy and print to screen
roc_auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'ROC AUC Score: {roc_auc:.2f}')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test)))

# Plot the heatmap to visualize our CM
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

