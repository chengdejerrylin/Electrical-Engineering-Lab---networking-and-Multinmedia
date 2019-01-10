import csv

list = []
with open('MNIST_train.csv', "r") as csv_file, open('MNIST_train_copy.csv', "a") as write_file:
    csv_reader = csv.reader(csv_file)
    counter = 0
    for row in csv_reader:
        if (counter == 0):
            counter = counter + 1
            continue
        last_element = row[-1]
        row = row[:-1]
        # print(int(float(last_element)))
        row.append("'"+str(int(float(last_element)))+"'")
        target = ",".join(row) + "\n"
        write_file.writelines(target)

#     print(csv_reader)
