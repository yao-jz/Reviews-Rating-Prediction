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
print("begin ada boost")
train_length = x_train.shape[0]

weights = np.array([1.0/train_length]*train_length)
for i in range(T):
	# 按照不同的weights概率sample出不同的样本训练集
	data_index = np.random.choice(train_length, size=train_length, p=weights)
	train_data = x_train[data_index]
	train_label = y_train[data_index]
	print("train tree for "+str(i))
	t = tree.DecisionTreeClassifier(class_weight='balanced')
	t.fit(train_data, train_label)
	train_pred = t.predict(train_data)
	e = float(np.dot(np.array(train_pred)!=np.array(train_label), weights))
	print("tree e = ", e)
	if e > 0.5:
		print("e > 0.5, break")
		break
	if e == 0.0:
		e += 0.00000001
	beta = e/(1.0-e)
	# 更新权重
	update_list = []
	for j in range(train_label.shape[0]):
		if train_label[j] != train_pred[j]:
			update_list.append(1)
		else:
			update_list.append(beta)
	weights = np.multiply(weights, update_list)
	# 归一化
	weights = weights / np.sum(weights)
	joblib.dump(t, "./model/tree/ada_tree_"+str(i)+".pkl")
	beta_file = open("./model/tree/ada_tree_beta_"+str(i)+".txt","w")
	beta_file.write(str(beta))
	beta_file.close()

weights = np.array([1.0/train_length]*train_length)
for i in range(T):
	# 按照不同的weights概率sample出不同的样本训练集
	data_index = np.random.choice(train_length, size=train_length, p=weights)
	train_data = x_train[data_index]
	train_label = y_train[data_index]
	print("train svm for "+str(i))
	svc = LinearSVC(class_weight="balanced", verbose=1)
	svc.fit(train_data, train_label)
	train_pred = svc.predict(train_data)
	e = float(np.dot(np.array(train_pred)!=np.array(train_label), weights))
	print("svm e = ", e)
	if e > 0.5:
		print("e > 0.5, break")
		break
	if e == 0.0:
		e += 0.00000001
	beta = e/(1.0-e)
	# 更新权重
	update_list = []
	for j in range(train_label.shape[0]):
		if train_label[j] != train_pred[j]:
			update_list.append(1)
		else:
			update_list.append(beta)
	weights = np.multiply(weights, update_list)
	# 归一化
	weights = weights / np.sum(weights)
	joblib.dump(svc, "./model/svm/ada_svm_"+str(i)+".pkl")
	beta_file = open("./model/svm/ada_svm_beta_"+str(i)+".txt","w")
	beta_file.write(str(beta))
	beta_file.close()
