import csv
def content(file_name, out_file):
    input_file = csv.reader(open(file_name, 'r', encoding='utf-8'))
    out = open(out_file, 'w', encoding='utf-8')
    for line in input_file:
        out.write(line[1])
        out.write('\n')