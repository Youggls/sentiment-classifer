import csv
filename = "./Sentiment Classification/data/furniture.csv"
data = csv.reader(open(filename, 'r', encoding='utf-8'))

for line in data:
    temp_aspect = line[2].split(';')
    temp_sentiment = line[3].split(';')
    temp_polarity = line[4].split(';')
    if len(temp_aspect) == 0 or len(temp_sentiment) == 0 or len(temp_polarity) == 0:
        print('The info is different in line {}'.format(line[0]))
    if (len(temp_aspect) != len(temp_sentiment)) or (len(temp_aspect) != len(temp_polarity)):
        print('The info is different in line {}'.format(line[0]))
    if '；' in line[2] or '；' in line[3] or '；' in line[4]:
        print('The info is different in line {}'.format(line[0]))
    if '‘' in line[2] or '‘' in line[3] or '‘' in line[4]:
        print('The info is different in line {}'.format(line[0]))
    if '’' in line[2] or '’' in line[3] or '’' in line[4]:
        print('The info is different in line {}'.format(line[0]))