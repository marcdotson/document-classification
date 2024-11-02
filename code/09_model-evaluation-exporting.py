#import neccessary libraies
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score
import csv
import os


# -----------------------------------------------------------------------------------------------
# Model Evaluation - Extended to Multiple Thresholds and export to CSV
# -----------------------------------------------------------------------------------------------


# Testing the best estimator on the test set based off of prior grid search or any best performing model
best_model = grid_search.best_estimator_  #ADD YOUR OWN MODEL HERE wiht the best parameters

#function that fits the model, displays evaluations, and exports results to a csv

def evaluate_model_then_export(
    model,
    X_train, y_train, X_test, y_test,
    stop_words_removed, input_data, input_type, sample_size_dist,
    thresholds=[0.5, 0.7, 0.9]
):
    """
    Logs model evaluation results with precision at specific thresholds to a CSV file.
    
    Parameters:
    - model: model object (RandomForestClassifier with model parameters)
    - X_train, y_train: Training data and labels
    - X_test, y_test: Test data and labels
    - stop_words_removed: bool, True if stop words were removed
    - input_data: string, data input type (e.g., "all text", "abstract")
    - input_type: string, input type with vectorization details or embedding type
    - sample_size_dist: string, sample size distribution (e.g., "var1: 100, var2: 600")
    - thresholds: list, probability thresholds to evaluate
    """
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Verify the order of the positive and negative classes
    print("Model class order:", model.classes_)
    
    # Get predicted probabilities for the positive class
    y_prob_test = model.predict_proba(X_test)[:, 1]
   
    
    # Define row data dictionary keys for CSV
    fieldnames = [
        'Model', 'Stop Words Removed', 'Input Data', 'Input Type',
        'Sample Size Distribution', 'Precision_50', 'Precision_70', 'Precision_90'
    ]
    
    # Prepare data row with general info and placeholders for threshold-specific columns
    row_data = {
        'Model': str(model),
        'Stop Words Removed': 'Yes' if stop_words_removed else 'No',
        'Input Data': input_data,
        'Input Type': input_type,
        'Sample Size Distribution': sample_size_dist
    }

    # Evaluate precision for each threshold and add it to row_data
    for threshold in thresholds:
        y_pred_custom = (y_prob_test >= threshold).astype(int)
        
        # Calculate precision
        precision = precision_score(y_test, y_pred_custom)
        
        # Add precision to the respective threshold column in row_data
        row_data[f'Precision_{int(threshold * 100)}'] = precision

        # Optional: Plot confusion matrix for the current threshold
        cm = confusion_matrix(y_test, y_pred_custom)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if threshold == 0.5 else "Oranges",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Threshold {threshold * 100:.0f}%')
        plt.show()

        print(f"Precision at {threshold * 100:.0f}%: {precision:.2f}")


     
    # Prepare EXCEL file for logging results
    excel_file = 'INSERT_YOUR_EXCEL_FILE_PATH_HERE.XLSX'

     # Create or append to the Excel file
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl') as writer:
            df = pd.DataFrame([row_data])
            df.to_excel(writer, sheet_name='Model Performance', index=False, header=not writer.sheets)
    else:
        df = pd.DataFrame([row_data])
        df.to_excel(excel_file, sheet_name='Model Performance', index=False)

    print(f"Results logged for model: {str(model)}")