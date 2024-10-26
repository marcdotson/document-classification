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

# Custom threshold, specify what you want it to be
custom_threshold = .8
y_pred_custom = (y_prob >= custom_threshold ).astype(int) 

#custom threshold classification report
print(f"\nClassification Report Custom threshold ({custom_threshold *100}%):")
print(classification_report(y_test, y_pred_custom))

# Calculate ROC AUC score and accuracy for default threshold
roc_auc = roc_auc_score(y_test, y_prob)
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f'\nAccuracy with default threshold (50%): {accuracy_default * 100:.2f}%')
print(f'ROC AUC Score: {roc_auc:.2f}')

# Calculate accuracy for custom threshold%
accuracy_cust = accuracy_score(y_test, y_pred_custom)
print(f'Accuracy with custom threshold ({custom_threshold *100}%): {accuracy_cust * 100:.2f}%')

# Compute confusion matrix for default threshold
cm_default = confusion_matrix(y_test, y_pred_default)
# Compute confusion matrix for custom threshold
cm_custom = confusion_matrix(y_test, y_pred_custom)

# Plot the heatmap to visualize our Confusion Matrix for the default threshold
plt.figure(figsize=(10, 7))
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Default Threshold (50%)')
plt.show()

# Plot the heatmap to visualize our Confusion Matrix for the custom threshold
plt.figure(figsize=(10, 7))
sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Custom Threshold ({custom_threshold *100}%)')
plt.show()

