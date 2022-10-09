# prediction
import math
from sklearn.svm import LinearSVC
import joblib
from sklearn import tree
from tqdm import tqdm
import random
import json
import numpy as np
from collections import defaultdict
T = 10
print("loading test data")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
result = {}
print("evaluating baseline svm")
model = joblib.load("./model/svm/baseline_svm.pkl")
r = model.predict(x_test)
np.save("./result/baseline_svm.npy", r)
print("evaluating baseline tree")
model = joblib.load("./model/tree/baseline_tree.pkl")
r = model.predict(x_test)
np.save("./result/baseline_tree.npy", r)

# 多数投票
print("evaluating bagging svm")
r = []
model_list = [joblib.load("./model/svm/bagging_svm_"+str(i)+".pkl") for i in range(T)]
for test in tqdm(x_test[:]):
	label = defaultdict(int)	
	for i in range(T):
		pred = model_list[i].predict(test.reshape(1, -1))
		label[pred[0]] += 1
	r.append(max(label.items())[0])
r=np.array(r)
np.save("./result/bagging_svm.npy",r)

print("evaluating bagging tree")
r = []
model_list = [joblib.load("./model/tree/bagging_tree_"+str(i)+".pkl") for i in range(T)]
for test in tqdm(x_test[:]):
	label = defaultdict(int)	
	for i in range(T):
		pred = model_list[i].predict(test.reshape(1, -1))
		label[pred[0]] += 1
	r.append(max(label.items())[0])
r=np.array(r)
np.save("./result/bagging_tree.npy",r)

print("evaluating ada boost svm")
r = []
beta_list = [float(open("./model/svm/ada_svm_beta_"+str(i)+".txt","r").read()) for i in range(T)]
model_list = [joblib.load("./model/svm/ada_svm_"+str(i)+".pkl") for i in range(T)]
for test in tqdm(x_test[:]):
	label = defaultdict(int)	
	for i in range(T):
		pred = model_list[i].predict(test.reshape(1, -1))
		label[pred[0]] += math.log(1.0/beta_list[i])
	r.append(max(label.items())[0])
r=np.array(r)
np.save("./result/ada_svm.npy",r)

print("evaluating ada boost tree")
r = []
beta_list = [float(open("./model/tree/ada_tree_beta_"+str(i)+".txt","r").read()) for i in range(T)]
model_list = [joblib.load("./model/tree/ada_tree_"+str(i)+".pkl") for i in range(T)]
for test in tqdm(x_test[:]):
	label = defaultdict(int)	
	for i in range(T):
		pred = model_list[i].predict(test.reshape(1, -1))
		label[pred[0]] += math.log(1.0/beta_list[i])
	r.append(max(label.items())[0])
r=np.array(r)
np.save("./result/ada_tree.npy",r)
