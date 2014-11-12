from math import ceil
import score


def split(table, coef=0.75, shuffle=False):
    if shuffle:
        table.shuffle()
    table_size = len(table)
    learning_size = int(table_size * coef)
    return table[:learning_size], table[learning_size:]


def k_fold(table, N, algo, columns, class_column, shuffle=False, score=score.f1_score):
    if shuffle:
        table.shuffle()
    step = int(ceil(float(len(table)) / N))
    result = 0.0
    for i in xrange(N):
        learning = table[:]
        test = learning[i * step:(i + 1) * step]
        del learning[i * step:(i + 1) * step]
        algo.learn(learning, columns, class_column)
        test_results = algo.test(test)
        result += score(*test_results)
    return result / N
