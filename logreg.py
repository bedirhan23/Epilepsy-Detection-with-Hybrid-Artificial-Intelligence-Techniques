import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

def specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def logreg_model(data, verbose=False):
    X = data.drop(columns=['label'], axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    logreg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100000)

    logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy


    roc_auc = roc_auc_score(y_test, logreg_model.predict_proba(X_test), multi_class='ovr')

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
    logreg_model(data)