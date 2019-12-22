
def stat():
    with open('basic.graph.train.tsv', 'r') as f:
        a = [0 for i in range(3778)]
        for line in f:
            id, _, _, _ = line[:-1].split('\t')
            id = int(id)
            a[id] = a[id] + 1
    print(a)
    print(len(a))
    b = [0 for i in range(20)]
    for n in a:
        if n > 10000:
            print(n)
        b[n / 1000] += 1
    print(b)

def collect_result():
    results = [[], [], [], [], [], [], [], []]
    with open('query_log_2', 'r') as f:
        for line in f:
            if not line.startswith('INFO:root:th-'):
                continue
            thid = int(line[13])
            ans = '\t'.join(line[:-1].split('\t')[1:])
            results[thid].append(ans)

    for res in results:
        print(len(res))
    results = sum(results, [])
    with open('sklt.txt', 'w') as f:
        for res in results:
            f.write(res)
            f.write('\n')


if __name__ == '__main__':
    collect_result()