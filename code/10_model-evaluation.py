#import neccessary libraies
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------------------------------------------------------------------------
# Model Evaluation - Extended to Multiple Thresholds
# -----------------------------------------------------------------------------------------------

# Define thresholds to evaluate
thresholds = [0.5, 0.7, 0.8, 0.9]  # Default threshold + other custom thresholds

# Testing the best estimator on the test set based off of prior grid search or any best performing model
best_model = grid_search.best_estimator_  #ADD YOUR OMDEL HERE AND FIT IT IF NEEDED
y_pred_default = best_model.predict(X_test)

# Verify the order of the positive and negative classes before referencing the positive class
print("Model class order:", best_model.classes_)

# Get predicted probabilities for the positive class
y_prob_test = best_model.predict_proba(X_test)[:, 1]  # Use [:, 1] for the positive class

# Evaluate for each threshold
for threshold in thresholds:
    y_pred_custom = (y_prob_test >= threshold).astype(int)
    
    # Classification report for the current threshold
    print(f"\nClassification Report for threshold {threshold * 100:.0f}%:")
    print(classification_report(y_test, y_pred_custom))
    
    # Accuracy for the current threshold
    accuracy = accuracy_score(y_test, y_pred_custom)
    print(f'Accuracy with threshold {threshold * 100:.0f}%: {accuracy * 100:.2f}%')
    
    # Compute and plot confusion matrix for the current threshold
    cm = confusion_matrix(y_test, y_pred_custom)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if threshold == 0.5 else "Oranges", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Threshold {threshold * 100:.0f}%')
    plt.show()
