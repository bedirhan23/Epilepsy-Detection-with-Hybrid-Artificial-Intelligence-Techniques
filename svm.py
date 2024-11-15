import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

def specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)

def svm_model(data, C=10, gamma=0.1, kernel='rbf', verbose=False):
    X = data.drop(columns=['label'], axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy

    if len(y.unique()) > 2:
        roc_auc = roc_auc_score(y_test, svm.predict_proba(X_test), multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

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

    return [accuracy, error, roc_auc, precision, recall, f1, spec]

if __name__ == "__main__":
    data = pd.read_csv("df_all_cases.csv")
    svm_model(data, verbose=True)
