from random import randint
from classifiers import knn_classifier
import ml_table
import score
from validation import k_fold


def create_set(size, a, b, div, name="table.csv"):
    table = ml_table.ml_table()
    for i in xrange(size):
        e = [randint(a, b), randint(a, b)]
        if e[0] < div:
            e.append(0)
        else:
            e.append(1)
        table.add_row(e, enforce_types=False)
    table.set_headers(["x", "y", "class"])
    table.enforce_types()
    table.save_table(name)
    return table

def test_classifier(table, k=5):
    knn = knn_classifier(k)
    result = k_fold(table, 4, knn, ["x", "y"], "class", score=score.auc_score, shuffle=False)
    return result

if __name__ == '__main__':
    table = create_set(100, 0, 100, 50)
    # table = ml_table.ml_table.load_table("table.csv", has_headers=True, shuffle=False)
    result = test_classifier(table)
    print result
