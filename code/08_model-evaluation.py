#import neccessary libraies
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score



# -----------------------------------------------------------------------------------------------
# Model Evaluation - Extended to Multiple Thresholds and print results to screen
# -----------------------------------------------------------------------------------------------

#ADD YOUR OWN MODEL HERE with the best parameters, if you used a grid search use this code otherewise specify your model
best_model = grid_search.best_estimator_  


#function that fits the model, displays evaluations, and stores our precision scores
def evaluate_model_store_precision( model, X_train, y_train, X_test, y_test,thresholds=[0.5, 0.7, 0.9]):
   
    # Fit the model
    model.fit(X_train, y_train)
    
    # Get predicted probabilities for the positive class
    y_prob_test = model.predict_proba(X_test)[:, 1]
   
   #create an empty dictionary to store our precision results
    precision_results = {}

    # Evaluate precision for each threshold and add it to row_data
    for threshold in thresholds:
        y_pred_custom = (y_prob_test >= threshold).astype(int)
        
        # Calculate precision
        precision = precision_score(y_test, y_pred_custom)
        
        # Add precision to the respective threshold column in row_data
        precision_results[f'Precision_{int(threshold * 100)}'] = precision

        # Plot confusion matrix for the current threshold
        cm = confusion_matrix(y_test, y_pred_custom)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if threshold == 0.5 else "Oranges",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Threshold {threshold * 100:.0f}%')
        plt.show()

        #print all of our results
        print(f"Precision at {threshold * 100:.0f}%: {precision:.2f}") 
        print(f"Classification Report at {threshold * 100:.0f}%:\n")
        print(classification_report(y_test, y_pred_custom, target_names=model.classes_))
        print("\n" + "-"*50 + "\n")


    return precision_results



