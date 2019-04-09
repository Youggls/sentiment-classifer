import csv
usr_dict = './Sentiment Classification/outdata/Elong.csv'
f = csv.reader(open(usr_dict, 'r', encoding = 'utf-8'))
temp = list()
for i in f:
    temp.append(i[0])
    temp.append(i[1])

temp = list(set(temp))
out = open('./Sentiment Classification/outdata/Elong-dic.txt', 'w', encoding='utf-8')
for i in temp:
    out.write(i)
    out.write('\n')