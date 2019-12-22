import codecs

def complete_candgen():
    root = "/home/xusheng/PythonProject/data/tabel"
    with codecs.open(root + "/size_116/candgen/candCell_50.txt.all", 'r', encoding='utf-8') as fin:
        dict_116 = dict()
        for line in fin:
            key = line.strip().split('\t')[0]
            dict_116[key] = line.strip()

    with codecs.open(root + "/size_200/candCell_50.txt.all", 'r', encoding='utf-8') as fin:
        dict_200 = dict()
        for line in fin:
            key = line.strip().split('\t')[0]
            dict_200[key] = line.strip()

    with codecs.open(root + "/size_200/candCell_50.txt.all.new", 'w', encoding='utf-8') as fout:
        for key, val in dict_200.items():
            if key not in dict_116:
                fout.write(val + "\n")
        for key, val in dict_116.items():
            fout.write(val + "\n")


if __name__ == "__main__":
    complete_candgen()
