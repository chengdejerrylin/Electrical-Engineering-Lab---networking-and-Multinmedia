from argparse import ArgumentParser
import csv

parser = ArgumentParser(description='A program that get the average of the eperiment data.')
parser.add_argument('inputFile', help='input csv file.')
parser.add_argument('-n', dest='nAcc',default=4, help='# of accuracy.')

args = parser.parse_args()
outputFile = ''.join(args.inputFile.split('.')[:-1]) + '_average.csv'

#output
isTitle = True
current = None
num = 0
acc = []
for i in range(args.nAcc) : acc.append(0.0)


with open(args.inputFile, 'r', newline='') as ifs :
	data = csv.reader(ifs)
	with open(outputFile, 'w', newline='') as os :
		writer = csv.writer(os, delimiter=',')

		for row in data :
			if isTitle :
				writer.writerow(row)
				isTitle = False
				continue

			if current is None : current = list(row)
			for i in range(len(row) - args.nAcc) :
				if current[i] != row[i] :
					if num != 0 :
						for j in range(args.nAcc) : current[-args.nAcc +j] = acc[j]/num
						print(current)
						writer.writerow(current)

						current = list(row)
						num = 0
						for j in range(args.nAcc) : acc[j] = 0.0
					break

			for i in range(args.nAcc) : acc[i] += float( row[-args.nAcc+i] )
			num += 1

		#end
		if num != 0 :
			for j in range(args.nAcc) : current[-args.nAcc +j] = acc[j]/num
			writer.writerow(current)

