# data shuffle
import random
csv_path = 'exp3-reviews.csv'
file = open(csv_path, "r").read().split("\n")
data_list = []
for i in file[1:]:
	l = i.split("\t")
	data_list.append(l)
data_list.pop(-1)
random.shuffle(data_list)
data_list = data_list[:44000]
out = open("source.txt", "w")
for i in data_list:
	out.write(i[5] + "\n")
out.close()
out = open("target.txt", "w")
for i in data_list:
	out.write(str(int(float(i[0]))) + "\n")
out.close()