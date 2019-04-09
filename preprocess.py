import csv

filename = "./Sentiment Classification/data/furniture.csv"
out = "./Sentiment Classification/outdata/furniture.csv"

aspect = list()
sentiment = list()
polarity = list()

data = csv.reader(open(filename, 'r', encoding='utf-8'))

for line in data:
    if len(line[2]) != 0 and len(line[3]) != 0 and len(line[4]) != 0:
        temp_aspect = line[2].split(';')
        temp_sentiment = line[3].split(';')
        temp_polarity = line[4].split(';')
        if (len(temp_aspect) != len(temp_sentiment)) or (len(temp_aspect) != len(temp_polarity)):
            continue
        else:
            for i in temp_aspect:
                aspect.append(i)
            for i in temp_polarity:
                polarity.append(i)
            for i in temp_sentiment:
                sentiment.append(i)

outfile = csv.writer(open(out, 'w', encoding = 'utf-8', newline = ''))

for i in range(len(aspect)):
    temp = list()
    temp.append(aspect[i])
    temp.append(sentiment[i])
    temp.append(polarity[i])
    outfile.writerow(temp)