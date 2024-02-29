# Multiclass Emotion Classification Project 

## Abstract 
Downloaded and extracted annotated data of six emotions from an online dataset. Used three models in Scikit-learn (Multinomial Naïve Bayes, Logistic Regression, and Random Forest) to train and test the data with three different features (bag of words, positive lexicon, negative lexicon) and three values for each hyperparameter (alpha, C, and n_estimators). A binary classification was also done to compare models' performances on different tasks. Reported the highest accuracy of 0.89 for multiclass classification and 0.95 for binary classification. Discussions on the result is provided at the end. 

## Dataset
The dataset is downloaded from Papers With Code, specifically from CARER (Contextualized Affect Representations for
Emotion Recognition). The authors selected a certain number of tweets from two widely used datasets and annotated the
data using distant supervision (Saravia et al., 2018). The downloaded file was pickled and transformed into a csv
through pandas for easier data processing. Table 1 below is an example of the data set. All words were separated by a
single white space. All texts were in lower case without punctuation or number. Under each number, there are 6 texts
corresponding to 6 different labels: joy, love, fear, surprise, anger, sadness. There are 14917 such sets of
instances (i.e., 14917*6 = 89502 instances in total).

|  Number  | Text                                                                                                                 | Label    |
|----------|----------------------------------------------------------------------------------------------------------------------|----------|
| 4        | a couple of years ago during the summer holiday                                                                      | joy      |
| 4        | i so many people in that guild the dumplings jass dodd bob john alll of my pallies that i feel so loyal to the guild | love     |
| 4        | a car came very close to hitting me whilst i was crossing the street                                                 | fear     |
| 4        | years ago i served in the army once a collegue denounced me because of a delict                                      | anger    |
| 4        | i a feeling of curious satisfaction to be on the same mission and a planetary co leader with tor                     | surprise |
| 4        | a breakup with someone i really liked                                                                                | sadness  |

Table 1: Example instances under number 4. 6 texts in the middle correspond to distinct emotions on the right.

The selected data was then copied into another csv file, with the label on the third column changed to “positive” and
“negative.” “love”, “surprise”, and “joy” were replaced by “positive”, and “fear”, “sadness”, and
“anger” were replaced by “negative.” Table 2 below provides the example of data with two labels. Numbers and texts stay
the same, but the labels are different from Table 1. The csv file with the six labels was used for multiclass
classification, and the one with two labels was used for binary classification.

| Number  | Text                                                                                                                 | Label     |
|---------|----------------------------------------------------------------------------------------------------------------------|-----------|
| 4       | a couple of years ago during the summer holiday                                                                      | positive  |
| 4       | i so many people in that guild the dumplings jass dodd bob john alll of my pallies that i feel so loyal to the guild | positive  |
| 4       | a car came very close to hitting me whilst i was crossing the street                                                 | negative  |
| 4       | years ago i served in the army once a collegue denounced me because of a delict                                      | negative  |
| 4       | i a feeling of curious satisfaction to be on the same mission and a planetary co leader with tor                     | positive  |
| 4       | a breakup with someone i really liked                                                                                | negative  |

Table 2: A set of example data under number 4. 6 texts in the middle correspond to only “positive” and “negative”.

When doing different tasks, both datasets were separated into training set, development set, and test set, by calling
the train_test_split method in Scikit-Learn twice. To keep the accuracy score deterministic, random state was set to 1.
There are 71865 instances in the training set, 8983 instances in the development set, and 8984 instances in the test
set.

## Sentiment Lexicon and Changes on Label

Positive and negative lexicon are two features used in this project. The lexicon was downloaded from Opinion Mining,
Sentiment Analysis, and Opinion Spam Detection (Liu and Hu, 2004). There are 4783 negative words and 2007 positive words
provided in two files. The files were transformed into csv for easier feature extraction.


## Development Set Results

There are two tasks involved in the project: binary classification and multiclass classification. Their development set
results are listed in Section 4.1 and Section 4.2 below. For each task, three features and three models were used – bag
of words, positive lexicon, and negative lexicon for features, Naïve Bayes, Logistic Regression, and Random Forest for
models. For each model, there are three values for hyperparameter tuning:alpha = (0.5, 1.0, 2.0), C = (0.5, 1.0, 2.0),
and n_estimators = (100, 200, 500).

### Binary Classification Results

Table 3 below includes the accuracy score for each configuration under the binary classification task. Across all
configurations, the one that gives the best accuracy was Logistic Regression with bag of words feature and C = 0.5.
Within Naïve Bayes, the highest accuracy is 94.43, generated from bag of words feature and alpha = 2.0. As for Random
Forest, the highest accuracy 93.67 also comes from bag of words feature, with n_estimator being 200 or 500, as they give
the exact same accuracy score.

|        Model        |     feature      |   hyperparameter   | accuracy  |
|:-------------------:|:----------------:|:------------------:|:---------:|
|     Naive Bayes     |   bag of words   |    alpha = 0.5     |   93.48   |
|                     |                  |    alpha = 1.0     |   94.00   |
|                     |                  |    alpha = 2.0     |   94.43   |
|                     | Positive lexicon |    alpha = 0.5     |   75.04   |
|                     |                  |    alpha = 1.0     |   73.76   |
|                     |                  |    alpha = 2.0     |   72.99   |
|                     | Negative Lexicon |    alpha = 0.5     |   62.03   |
|                     |                  |    alpha = 1.0     |   61.75   |
|                     |                  |    alpha = 2.0     |   61.49   |
| Logistic Regression |   bag of words   |      C = 0.5       | **95.85** |
|                     |                  |      C = 1.0       |   95.77   |
|                     |                  |      C = 2.0       |   95.67   |
|                     | Positive lexicon |      C = 0.5       |   75.39   |
|                     |                  |      C = 1.0       |   75.31   |
|                     |                  |      C = 2.0       |   75.30   |
|                     | Negative Lexicon |      C = 0.5       |   83.55   |
|                     |                  |      C = 1.0       |   83.49   |
|                     |                  |      C = 2.0       |   83.39   |
|    Random Forest    |   bag of words   | n_estimators = 100 |   93.05   |
|                     |                  | n_estimators = 200 |   93.67   |
|                     |                  | n_estimators = 500 |   93.67   |
|                     | Positive lexicon | n_estimators = 100 |   53.75   |
|                     |                  | n_estimators = 200 |   53.73   |
|                     |                  | n_estimators = 500 |   53.73   |
|                     | Negative Lexicon | n_estimators = 100 |   83.31   |
|                     |                  | n_estimators = 200 |   83.37   |
|                     |                  | n_estimators = 500 |   83.34   |

Table 3: Accuracy scores for different configurations in binary classification.

### Multiclass Classification Results

Table 4 below includes the accuracy score for each configuration under the multiclass classification task. Across all
configurations, the one that gives the best accuracy is again Logistic Regression with bag of words feature and C = 0.5.
Within Naïve Bayes, the highest accuracy is 86.67, coming from bag of words feature and alpha = 2.0. Within Random
Forest, the highest accuracy 86.17 also comes from bag of words feature, with n_estimator being 500.

|        Model        |     feature      |   hyperparameter   | accuracy  |
|:-------------------:|:----------------:|:------------------:|:---------:|
|     Naive Bayes     |   bag of words   |    alpha = 0.5     |   85.22   |
|                     |                  |    alpha = 1.0     |   86.11   |
|                     |                  |    alpha = 2.0     |   86.67   |
|                     | Positive lexicon |    alpha = 0.5     |   43.03   |
|                     |                  |    alpha = 1.0     |   43.03   |
|                     |                  |    alpha = 2.0     |   42.80   |
|                     | Negative Lexicon |    alpha = 0.5     |   49.36   |
|                     |                  |    alpha = 1.0     |   49.35   |
|                     |                  |    alpha = 2.0     |   49.11   |
| Logistic Regression |   bag of words   |      C = 0.5       | **89.68** |
|                     |                  |      C = 1.0       |   89.25   |
|                     |                  |      C = 2.0       |   88.61   |
|                     | Positive lexicon |      C = 0.5       |   43.86   |
|                     |                  |      C = 1.0       |   43.72   |
|                     |                  |      C = 2.0       |   43.64   |
|                     | Negative Lexicon |      C = 0.5       |   60.49   |
|                     |                  |      C = 1.0       |   60.33   |
|                     |                  |      C = 2.0       |   60.28   |
|    Random Forest    |   bag of words   | n_estimators = 100 |   85.28   |
|                     |                  | n_estimators = 200 |   85.85   |
|                     |                  | n_estimators = 500 |   86.17   |
|                     | Positive lexicon | n_estimators = 100 |   20.28   |
|                     |                  | n_estimators = 200 |   20.26   |
|                     |                  | n_estimators = 500 |   20.25   |
|                     | Negative Lexicon | n_estimators = 100 |   21.57   |
|                     |                  | n_estimators = 200 |   21.55   |
|                     |                  | n_estimators = 500 |   20.24   |

Table 4: Accuracy scores for different configurations in multiclass classification

## Test Set Results

For both tasks, the configurations that resulted in the highest accuracy in each model with development set data were
applied on the test set.
Section 5.1 and Section 5.2 will each provide two tables related to test set results for binary classification and
multiclass classification
– the accuracy scores with the test data and the classification report for the configuration with the highest accuracy
score generated from the test set.

### Binary Classification Test Set Results

Table 5 shows the accuracy with the highest configurations in development set for each model. As suggested in Section
4.1, the feature that gives the highest accuracy across all three models is bag of words. With alpha = 2.0 for Naïve
Bayes, C = 0.5 for Logistic Regression, and n_estimators = 200 for Random Forest, we have the accuracy of 93.78, 95.74,
93.42. Therefore, Logistic Regression model with C = 0.5 and bag of words feature returns the highest accuracy score for
binary classification. The classification report of this configuration implemented on the test set is provided in Table

|        Model        |   feature    |  hyperparameter   | accuracy |
|:-------------------:|:------------:|:-----------------:|:--------:|
|     Naïve Bayes     | bag of words |    alpha = 2.0    |  93.78   |
| Logistic Regression | bag of words |      C = 0.5      |  95.74   |
|    Random Forest    | bag of words | n_estimator = 200 |  93.42   |

Table 5: Accuracy scores with test data with the configurations contributing to the highest accuracy scores with the
development set for binary classification

|              | Precision  | Recall  | F1-Score | Support  |
|--------------|------------|---------|----------|----------|
| Negative     | 0.96       | 0.95    | 0.96     | 4486     |
| Positive     | 0.95       | 0.96    | 0.96     | 4498     |
|              |            |         |          |          |
| Accuracy     |            |         | 0.96     | 8984     |
| Macro avg    | 0.96       | 0.96    | 0.96     | 8984     |
| Weighted avg | 0.96       | 0.96    | 0.96     | 8984     |

Table 6: Classification report for the best configuration: Logistic Regression model, C = 0.5, bag of words features,
binary classification

### Multiclass Classification Test Set Results

According to Section 4.2, the highest accuracy scores for each model all have bag of words as the feature. With the
hyperparameters setting to alpha = 2.0, C = 0.5, n_estimators = 500, we have the accuracy of 85.34, 89.77, 86.10 from
three different models, as seen in Table 7 below. Thus, the classification report of the configuration that contributed
to the score of 90, Logistic regression model with C = 0.5 and bag of words features, is provided here as Table 8.

|        Model        |   feature    |  hyperparameter   | accuracy |
|:-------------------:|:------------:|:-----------------:|:--------:|
|     Naïve Bayes     | bag of words |    alpha = 2.0    |  85.34   |
| Logistic Regression | bag of words |      C = 0.5      |  89.77   |
|    Random Forest    | bag of words | n_estimator = 500 |  86.10   |

Table 7: Accuracy scores with test data resulted from the configurations that contributed to the highest accuracy scores
with the development set for multiclass classification

|              | Precision  | Recall  | F1-Score | Support  |
|--------------|------------|---------|----------|----------|
| anger        | 0.88       | 0.90    | 0.89     | 1520     |
| fear         | 0.87       | 0.83    | 0.85     | 1466     |
| joy          | 0.89       | 0.86    | 0.88     | 1404     |
| love         | 0.92       | 0.94    | 0.93     | 1529     |
| sadness      | 0.92       | 0.89    | 0.91     | 1500     |
| surprise     | 0.90       | 0.95    | 0.93     | 1565     |
|              |            |         |          |          |
| Accuracy     |            |         | 0.90     | 8984     |
| Macro avg    | 0.90       | 0.90    | 0.90     | 8984     |
| Weighted avg | 0.90       | 0.90    | 0.90     | 8984     |

Table 8: Classification report for the best configuration: Logistic Regression model, C = 0.5, bag of words features,
multiclass classification

## Discussion

There are some noticeable trends in both classification tasks, even though accuracy scores in binary classification are
higher than those in multiclass classification. Across the features, we can see that bag of words outperforms positive
lexicon and negative lexicon. This can be seen from the fact that the highest configurations in the development set
always have bag of words feature. This is likely because that for bag of words, every word in the text can be used by
the model to learn and decide the text’s label, while there are only around 4700 words in negative lexicon and 2000
words in positive lexicon, so models are learning and deciding with limited information, and thus predict less
accurately. This can also explain why the accuracy from positive lexicon features is generally lower than that from
negative ones – there is a 2700 difference between the number of words in the two lists. Although this contradicts with
the assumption that sentiment lexicon will perform better than bag of words, as bag of words is a word-level feature
while sentiment lexicon deals with the semantic meaning behind the words, the huge difference between the number of
features makes the finding reasonable.

Another explanation for this finding is that the sentiment lexicon might mis-extract some features. As shown in Table 1
and 2, there are some words not spelt correctly, such as “collegue” and “alll”, since the texts are tweets in real life.
However, sentiment lexicon only has correctly spelt words. Therefore, when extracting features in the texts, some
sentiment words may be misidentified as not a feature due to misspelling, thus contributing to lower accuracy.

Thanks to the large and balanced dataset, all three models performed well. With the same features, there is no
significant difference between their accuracy score. Nevertheless, Logistic Regression slightly outperforms Naïve Bayes
and Random Forest, which can be seen from the fact that the two highest configurations on the test set both have
Logistic Regression as the model. This might be because that data is good enough for the model to find a clear optimal
decision boundary for each label, or a clear relationship between features and labels. It may also be because Logistic
Regression is generally a better model for text classification than Naïve Bayes and Random Forest.

Since for each model there are only three values used for hyperparameter tuning, it is likely that there are higher
accuracy scores than the one reported here that the hyperparameters used here cannot generate. If that is the case, then
the comparisons and reason behind the findings stated above may not be true.
