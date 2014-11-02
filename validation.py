from math import ceil


def split(table, coef=0.75, shuffle=False):
    if shuffle:
        table.shuffle()
    table_size = len(table)
    learning_size = int(table_size * coef)
    return table[:learning_size], table[learning_size:]


def k_fold(table, N, algo, shuffle=False):
    if shuffle:
        table.shuffle()
    step = int(ceil(float(len(table)) / N))
    result = 0.0
    for i in xrange(N):
        learning = table[:]
        test = learning[i * step:(i + 1) * step]
        del learning[i * step:(i + 1) * step]
        algo.learn(learning)
        result += algo.test(test)
    return result / N
