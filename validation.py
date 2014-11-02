from math import ceil
import ml_table


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
        learning_rows = table.get_rows()
        test_rows = learning_rows[i * step:(i + 1) * step]
        del learning_rows[i * step:(i + 1) * step]
        learning = ml_table.ml_table(rows=learning_rows, headers=table.get_headers())
        test = ml_table.ml_table(rows=test_rows, headers=table.get_headers())
        algo.learn(learning)
        result += algo.test(test)
    return result / N
