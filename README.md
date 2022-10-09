# Reviews-Rating-Prediction

Using an ensemble learning approach, bagging and adaboost algorithms are implemented to perform the classification.

## Implementation Details

### Feature Extraction

Using the tfidf vectorization method to do the feature extraction for the content part of each data to obtain a vector for each comment text representation, and the vector is saved as an npy file for svm and decision tree training.

### Model

SVM and decision tree, using LinearSVC and decision tree models provided by sklearn for training and prediction respectively.

### Ensemble Learning

I implemented the Bagging and Ada Boost algorithm.

## Experiments

| Model  |MAE|RMSE|MSE|
|---|---|---|---|
|Baseline SVM|0.6015|1.074011173126239|1.1535|
|Baseline DTree|0.9195|1.4281106399715675|2.0395|
|Bagging SVM|**0.56725**|**1.0560539758932779**|**1.11525**|
|Bagging DTree|0.72275|1.308147545195113|1.71125|
|Ada Boost SVM|0.69325|1.2687592364195817|1.60975|
|Ada Boost DTree|0.72275|1.3085297092538632|1.71225|

