from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score, recall_score, precision_score
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np

y_test = np.load("y_test.npy")
bs = list(np.load("./result/baseline_svm.npy"))
bt = list(np.load("./result/baseline_tree.npy"))
bas = list(np.load("./result/bagging_svm.npy"))
bat = list(np.load("./result/bagging_tree.npy"))
adas = list(np.load("./result/ada_svm.npy"))
adat = list(np.load("./result/ada_tree.npy"))
result = [bs, bt, bas, bat, adas, adat]


# for r in result:
# 	print(mean_absolute_error(y_test, r),end="|")
# 	print(sqrt(mean_squared_error(y_test,r)),end="|")
# 	print(mean_squared_error(y_test, r),end="|")
# 	print(np.mean(np.abs((y_test-r)/y_test)))



for r in result:
	print("|".join([str(i)[:5] for i in list(precision_score(y_test, r, average=None))]))