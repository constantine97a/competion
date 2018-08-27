import numpy
import pandas as pd

answer = pd.read_csv(u'result.csv', skiprows=1)

output = pd.read_csv('output.txt')

csv_rows_tuples = zip(output.values, answer.values)
sum_result = 0.0

count = 1
for item in csv_rows_tuples:
    count += 1
    sum_result += numpy.square(float(item[1][-1]) - float(item[0][0]))

print ("loss of mean-square  is %s" % (sum_result / count))
print ("loss of root mean-square is %s" % numpy.sqrt(sum_result / count))
