from matplotlib.pyplot import *
from IPython.display import *
import numpy as np
import pickle, gzip
import csv


f = gzip.open('../mnist.pkl.gz','rb')
train_set, valid_set, test_set = pickle.load(f,encoding='unicode-escape')
f.close()
x_train,y_train=train_set[0],train_set[1]
x_valid,y_valid=valid_set[0],valid_set[1]
x_test,y_test=test_set[0],test_set[1]

# Examine the dataset:
print("Shape : ", x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print("Max and min of x_train and y_train : ", np.min(x_train),np.max(x_train),np.min(y_train),np.max(y_train))

# write testing answer to file
lister = []
with open("MNIST_test_tmp.csv", "a") as f:
    for i in range(0,784):
        lister.append("Field_"+str(i+1))
    lister.append("Answer")
    wr = csv.writer(f)
    wr.writerow(lister)
    for i in range(0,10000):
        tmp = x_test[i].tolist()
        tmp.append("'" + str(y_test[i]) + "'")
        wr.writerow(tmp)
# writer.writerows(x_test)






    # lister.append("Field_"+str(i+1))



# with open("MNIST_test_ans.csv", "a") as f:
#     f.write("Answer\n")
#     for item in y_test:
#         f.write("%s\n" % item)

# lister = []
# with open("MNIST_test.csv", "a") as f:
#     for i in range(0,784):
#         lister.append("Field_"+str(i+1))
#     lister.append("Answer")
#     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#     wr.writerow(lister)
#     writer = csv.writer(f)
#     writer.writerows(x_test)

# print(x_train[0])
# print(y_train[0])
# for i in range(50):
#     subplot(5,10,i+1)
#     imshow(x_test[i].reshape([28,28]),cmap='gray');
#     title(str(y_test[i]));
#     axis('off')
    
# Transform NumPy arrays to PyTorch tensors:
# X_train=t.from_numpy(x_train)
# Y_train=t.from_numpy(y_train)
# X_test=t.from_numpy(x_test)
# Y_test=t.from_numpy(y_test)

# # print(X_train)