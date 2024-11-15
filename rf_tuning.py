import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)

def random_forest_model(data, verbose=False):
    X = data.drop(columns=['label'], axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                               param_grid=param_grid,
                               cv=5, verbose=verbose, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    if verbose:
        print(f"Best Parameters: {best_params}")

    best_rf_model = grid_search.best_estimator_

    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy

    # Handle multi-class case for roc_auc_score
    if len(y.unique()) > 2:
        roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test), multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    spec = specificity(y_test, y_pred)

    if verbose:
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy}")
        print(f"Error: {error}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Specificity: {spec}")

    return [accuracy, error, roc_auc, precision, recall, f1, spec, best_params]

if __name__ == "__main__":
    data = pd.read_csv("df_all_cases.csv")
    random_forest_model(data, verbose=True)
