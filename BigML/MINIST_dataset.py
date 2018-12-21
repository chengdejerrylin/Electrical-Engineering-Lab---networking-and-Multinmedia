from matplotlib.pyplot import *
from IPython.display import *
import numpy as np
import pickle, gzip
import csv


f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = pickle.load(f,encoding='unicode-escape')
f.close()
x_train,y_train=train_set[0],train_set[1]
x_valid,y_valid=valid_set[0],valid_set[1]
x_test,y_test=test_set[0],test_set[1]

# Examine the dataset:
print("Shape : ", x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print("Max and min of x_train and y_train : ", np.min(x_train),np.max(x_train),np.min(y_train),np.max(y_train))


# with open("MNIST.csv", "w") as f:
#     for i in range(0, len(y_train)):
#         new_train = np.append(x_train[i], y_train[i])
#         writer = csv.writer(f)
#         writer.writerow(new_train)



with open("MNIST_test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x_test)



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