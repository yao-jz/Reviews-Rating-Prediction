from sklearn.svm import LinearSVC
import joblib
from sklearn import tree
from tqdm import tqdm
import random
import numpy as np
T=10
print("loading data")
x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
print("start bagging...")
train_length = x_train.shape[0]
for i in tqdm(range(T)):
	sub_x_train = []
	sub_y_train = []
	for j in range(train_length):
		index = int(random.random()*train_length)
		sub_x_train.append(x_train[index])
		sub_y_train.append(y_train[index])
	sub_x_train = np.array(sub_x_train)
	sub_y_train = np.array(sub_y_train)
	print("training svm for " + str(i))
	svc = LinearSVC(class_weight="balanced", verbose=1)
	svc.fit(sub_x_train, sub_y_train)
	joblib.dump(svc, "./model/svm/bagging_svm_"+str(i)+".pkl")
	print("training tree for " + str(i))
	t = tree.DecisionTreeClassifier(class_weight='balanced')
	t.fit(sub_x_train, sub_y_train)
	joblib.dump(t, "./model/tree/bagging_tree_"+str(i)+".pkl")