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
y_pred_default = best_model.predict(X_test)

# Verify the order of the positive and negative classes before referencing the positive class
print("Model class order:", best_model.classes_)

# Get predicted probabilities for the positive class; make sure the index aligns with the class order above
y_prob = best_model.predict_proba(X_test)[:, 1]  # Use [:, 1] for the positive class

# Default threshold of 50%
print("\nClassification Report Default threshold 50%:")
print(classification_report(y_test, y_pred_default))

# Custom threshold of 70%
y_pred_70 = (y_prob >= 0.7).astype(int)  # Using 0.7 as threshold

print("\nClassification Report Custom threshold 70%:")
print(classification_report(y_test, y_pred_70))

# Calculate ROC AUC score and accuracy for default threshold
roc_auc = roc_auc_score(y_test, y_prob)
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f'\nAccuracy with default threshold (50%): {accuracy_default * 100:.2f}%')
print(f'ROC AUC Score: {roc_auc:.2f}')

# Calculate accuracy for custom threshold of 70%
accuracy_70 = accuracy_score(y_test, y_pred_70)
print(f'Accuracy with custom threshold (70%): {accuracy_70 * 100:.2f}%')

# Compute confusion matrix for default threshold
cm_default = confusion_matrix(y_test, y_pred_default)
# Compute confusion matrix for custom threshold
cm_70 = confusion_matrix(y_test, y_pred_70)

# Plot the heatmap to visualize our Confusion Matrix for the default threshold
plt.figure(figsize=(10, 7))
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Default Threshold (50%)')
plt.show()

# Plot the heatmap to visualize our Confusion Matrix for the custom threshold
plt.figure(figsize=(10, 7))
sns.heatmap(cm_70, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Custom Threshold (70%)')
plt.show()

