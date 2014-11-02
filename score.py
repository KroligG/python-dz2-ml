def f1_score(matrix, test_set):
    [total, [tp, fp], [fn, tn]] = matrix
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def auc_score(matrix, test_set):
    class0 = filter(lambda x: x["class"] == 0, test_set)
    class1 = filter(lambda x: x["class"] == 1, test_set)
    auc = float(sum((int(e0["p"] > e1["p"]) for e0 in class0 for e1 in class1))) / (len(class0) * len(class1))
    return auc