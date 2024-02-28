import csv
from typing import Generator, Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def load_text(fname: str) -> Generator[Dict[str, str], None, None]:
    with open(fname, "r", encoding="utf8") as file:
        csvfile = csv.reader(file)
        list_of_data = list(csvfile)
        for data in list_of_data[1:]:
            label = data[2]
            text = data[1]
            yield {"label": label, "text": text}


train_rows = list(load_text("data_2.csv"))  #data_2.csv has all text with positive and negative label
#train_rows = list(load_text("data_6.csv"))  #data_6.csv contains all text with six labels
y = np.array([row['label'] for row in train_rows])
bow_train = CountVectorizer(binary=True)
X = bow_train.fit_transform([row["text"] for row in train_rows])
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=1)

possible_n_estimators = [100, 200, 500]
accuracy_list = []
for estimator in possible_n_estimators:
    rfc = RandomForestClassifier(n_estimators=estimator)
    rfc.fit(X_train, y_train)
    y_pred_dev = rfc.predict(X_dev)
    accuracy_list.append(accuracy_score(y_dev, y_pred_dev))
    print(f"the accuracy for {estimator} is {accuracy_score(y_dev, y_pred_dev)}")

highest_accuracy = max(accuracy_list)
highest_n_estimator = possible_n_estimators[accuracy_list.index(max(accuracy_list))]
print("highest accuracy for Random Forest with bag of words features:", highest_accuracy)
print("the corresponding n_estimator is", highest_n_estimator)

rfc_test = RandomForestClassifier(n_estimators=highest_n_estimator)
rfc_test.fit(X_train, y_train)
y_pred_test = rfc_test.predict(X_test)
print("the accuracy in the test set is", accuracy_score(y_test, y_pred_test))
print("Classification report on the test set is below:")
print(classification_report(y_test, y_pred_test))

