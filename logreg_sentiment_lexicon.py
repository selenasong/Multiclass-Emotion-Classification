import csv
from typing import Generator, Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


data_rows = list(load_text("data_2.csv")) #data_2.csv contains all text with two labels. Call when doing binary classificaiton 
#data_rows = list(load_text("data_6.csv")) #data_6.csv contains all text with six labels. Call when doing multiclass classification


def load_sentiment_lexicon(fname: str):
    with open(fname, "r", encoding="utf8") as file:
        csvfile = csv.reader(file)
        list_of_data = list(csvfile)
        for data in list_of_data[1:]:
            yield (data[0], data[1])


pos_words = []
for row in load_sentiment_lexicon("sentiment_lexicon_positive.csv"):
    pos_words.append(row[0])
#sentiment_lexicon_positive.csv includes all positive sentiment words downloaded online

#neg_words = []
#for row in load_sentiment_lexicon("sentiment_lexicon_negative.csv"):
#    neg_words.append(row[0])
#sentiment_lexicon_negative.csv includes all negative sentiment words downloaded online


def pos_word_feature_counter(data):
    pos_count_dict = defaultdict(float)
    for word in data.split():
        if word in pos_words:
            pos_count_dict[word] = 1.0
        else:
            pos_count_dict[word] = 0.0
    return pos_count_dict

def neg_word_feature_counter(data):
    neg_count_dict = defaultdict(float)
    for word in data.split():
        if word in neg_words:
            neg_count_dict[word] = 1.0
        else:
            neg_count_dict[word] = 0.0
    return neg_count_dict

#positive lexicon feature transformer, comment out when running negative sentiment 
y = np.array([row['label'] for row in data_rows])
vectorizer = DictVectorizer()
X = vectorizer.fit_transform([pos_word_feature_counter(row["text"]) for row in data_rows])
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=1)


#negative lexicon feature transformer, comment out when running positive sentiment 
#y = np.array([row['label'] for row in data_rows])
#vectorizer = DictVectorizer()
#X = vectorizer.fit_transform([neg_word_feature_counter(row["text"]) for row in data_rows])
#X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.2, random_state=1)
#X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5, random_state=1)


possible_C = [0.5, 1.0, 2.0]
accuracy_list = []
for C in possible_C:
    logreg = LogisticRegression(C=C, max_iter=90000).fit(X_train, y_train)
    y_pred_dev = logreg.predict(X_dev)
    accuracy_list.append(accuracy_score(y_dev, y_pred_dev))
    print(f"the accuracy for {C} is {accuracy_score(y_dev, y_pred_dev)}")

highest_accuracy = max(accuracy_list)
highest_C = possible_C[accuracy_list.index(max(accuracy_list))]
print("highest accuracy for Logistic Regression with positive lexicon features:", highest_accuracy)
#print("highest accuracy for Logistic Regression with negative lexicon features:", highest_accuracy)
print("the corresponding C is", highest_C)

logreg_test = LogisticRegression(C=highest_C, max_iter=90000).fit(X_train, y_train)
y_pred_test = logreg_test.predict(X_test)
print("accuracy score on the test set is:", accuracy_score(y_test,y_pred_test))
print("Classification report on the test set is below:")
print(classification_report(y_test, y_pred_test))
