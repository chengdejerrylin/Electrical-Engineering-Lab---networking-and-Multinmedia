import csv

list = []
with open('data/datasets/MNIST/MNIST_train__title.csv', "a") as csv_file:
    for i in range(0,784):
        list.append("Field_"+str(i+1))
    list.append("Answer")
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    wr.writerow(list)

