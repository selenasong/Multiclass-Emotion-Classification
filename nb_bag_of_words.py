import csv
from typing import Generator, Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def load_text(fname: str) -> Generator[Dict[str, str], None, None]:
    with open(fname, "r", encoding="utf8") as file:
        csvfile = csv.reader(file)
        list_of_data = list(csvfile)
        for data in list_of_data[1:]:
            label = data[2]
            text = data[1]
            yield {"label": label, "text": text}

train_rows = list(load_text("data_6.csv"))  #data_6.csv contains all text with six labels
#train_rows = list(load_text("data_2.csv"))  #data_2.csv has all text with positive and negative label
y = np.array([row['label'] for row in train_rows])
bow_train = CountVectorizer(binary=True)
X = bow_train.fit_transform([row["text"] for row in train_rows])
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=1)

possible_k = [0.5, 1.0, 2.0]
accuracy_list = []
for k in possible_k:
    mnb = MultinomialNB(alpha=k)
    mnb.fit(X_train, y_train)
    y_pred_dev = mnb.predict(X_dev)
    accuracy_list.append(accuracy_score(y_dev, y_pred_dev))
    print(f"the accuracy for {k} is {accuracy_score(y_dev, y_pred_dev)}")

highest_accuracy = max(accuracy_list)
highest_k = possible_k[accuracy_list.index(max(accuracy_list))]
print("highest accuracy for Naive Bayes with bag of words features:", highest_accuracy)
print("the corresponding k is", highest_k)

mnb_test = MultinomialNB(alpha=highest_accuracy)
mnb_test.fit(X_train, y_train)
y_pred_test = mnb_test.predict(X_test)
print("accuracy score on the test set is:", accuracy_score(y_test,y_pred_test))
print("Classification report on the test set is below, with the highest configuration implemented:")
print(classification_report(y_test, y_pred_test))

