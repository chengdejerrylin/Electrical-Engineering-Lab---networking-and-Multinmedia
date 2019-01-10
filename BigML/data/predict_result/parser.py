import csv


def main():
    pass

def parser(model_name, input_number, output_category):
    with open(model_name + '/' + model_name + "_results") as readcsv:
        read_file = csv.reader(readcsv, delimiter=',')
        k = 0
        input_list = []
        answer_list = []
        output_list = []
        final_list = []
        for row in read_file:
            k = k+1
            input_list.append(row[0:int(input_number)])
            answer_list.append(row[int(input_number)])
            output_list.append(row[(int(input_number)+1) : (int(input_number) + int(output_category)+1)])
            final_list.append(input_list)
            final_list.append(answer_list)
            final_list.append(output_list)
            print(final_list)
            if k > 2:
                break

if __name__ == "__main__":
    model_name = input("Please input model name! ")
    input_number = input("Ur input variable number: ")
    output_category = input("Ur output variable category number: ")
    parser(model_name, input_number, output_category)
    
