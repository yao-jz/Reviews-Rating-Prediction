# baseline
from sklearn.svm import LinearSVC
import joblib
from sklearn import tree
import numpy as np

print("loading data")
x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("begin training baseline svm")
svc = LinearSVC(class_weight="balanced", verbose=1)
svc.fit(x_train, y_train)
print("saving model...")
joblib.dump(svc, "./model/svm/baseline_svm.pkl")

print("begin training baseline tree")
t = tree.DecisionTreeClassifier(class_weight='balanced')
t.fit(x_train, y_train)
print("saving model...")
joblib.dump(t, "./model/tree/baseline_tree.pkl")
