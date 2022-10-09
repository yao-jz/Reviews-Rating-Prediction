# 处理数据
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
source = open("source.txt", "r").read().split("\n")
target = open("target.txt","r").read().split("\n")
print(len(source)) # 44000
train_list = source[:40000]
test_list = source[40000:]
c0 = CountVectorizer()
call = c0.fit_transform([i for i in train_list+test_list])
c1 = CountVectorizer(vocabulary=c0.vocabulary_)
train = c1.fit_transform([i for i in train_list])
print("train shape", repr(train.shape))
c2 = CountVectorizer(vocabulary=c0.vocabulary_)
test = c2.fit_transform([i for i in test_list])
print("test shape", repr(test.shape))
print("start process data...")
tfidftransformer = TfidfTransformer()
x_train = tfidftransformer.fit_transform(train).toarray()
x_test = tfidftransformer.fit_transform(test).toarray()
print("process data over")
y_train = np.array([int(i) for i in target[:40000]])
y_test = np.array([int(i) for i in target[40000:]])
print("saving data")
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)