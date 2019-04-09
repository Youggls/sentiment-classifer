import csv
import random
f = csv.reader(open('./Sentiment Classification/outdata/make-up.csv', 'r', encoding='utf-8'))
writer = csv.writer(open('./Sentiment Classification/outdata/make-up-remove.csv', 'w', encoding='utf-8', newline=''))

for i in f:
    if i[2] == '1':
        if random.random() > 0.5:
            writer.writerow(i)
    else:
        writer.writerow(i)