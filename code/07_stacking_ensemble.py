
# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

  
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression())) #EXAMPLES ONLY RIGHT NOW
	level0.append(('rf', RandomForestClassifier(n_estimators=100)))  # Random Forest Classifier
	level0.append(('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)))  # XGBClassifier
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict() #EXAMPLES ONLY RIGHT NOW
	models['lr'] = LogisticRegression()
	models['xgb'] = KNeighborsClassifier()
	models['rf'] = RandomForestClassifier()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 
# define dataset #SPLIT HERE
## Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# get the models to evaluate
#Change this up
#Some are encoding, some are valuecounts?
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))






##################################################### SIMPLIFIED VERSION
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

# Split the data (assuming X and y are your features and labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the base level models (level0)
level0 = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False))
]

# Create the final estimator (level1)
level1 = LogisticRegression()

# Create the stacking ensemble with the level0 models and level1 model
stacking_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# Define hyperparameter grid for the final estimator (level1)
param_grid = {
    'final_estimator__C': [0.1, 1, 10],  # Regularization strength for LogisticRegression
    'final_estimator__solver': ['lbfgs', 'liblinear']  # Solver options for LogisticRegression
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)

# Fit the grid search to find the best parameters for the level1 model
grid_search.fit(X_train, y_train)

# Best parameters and best score for the final estimator
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters for Level1 (final estimator): {best_params}')
print(f'Best Cross-Validation Score for Level1: {best_score:.2f}')

# Train the final model with the best parameters
best_stacking_model = grid_search.best_estimator_

# Make predictions on the test data
y_pred = best_stacking_model.predict(X_test)
###########################################################################