from math import sqrt

def _euclid(x, y):
    s = map(lambda a, b: (a - b) ** 2, x, y)
    return sqrt(sum(s))

class knn_classifier:
    def __init__(self, k, distance=_euclid, threshold=0.5):
        self.k = k
        self.distance = distance
        self.threshold = threshold

    def learn(self, learn_set, columns, class_column):
        self.learn_set = learn_set
        self.class_column = class_column
        self.class_column_index = learn_set.get_index(class_column)
        if columns:
            self.columns = []
            for column in columns:
                self.columns.append(learn_set.get_index(column))
        else:
            self.columns = filter(lambda x: x != self.class_column_index, range(len(learn_set)))

    def test(self, test_set):
        [[tp, fp], [fn, tn]] = [[0, 0], [0, 0]]
        test_result = []
        for t in test_set:
            c, p = self.classify(t)
            real_class = t[self.class_column_index]
            test_result.append({"e": t, "p": p, "c": c, "class": real_class})
            if real_class == 0 and c == 0:
                tp += 1
            elif real_class == 1 and c == 0:
                fp += 1
            elif real_class == 0 and c == 1:
                fn += 1
            elif real_class == 1 and c == 1:
                tn += 1
        return [[tp, fp], [fn, tn]], test_result

    def classify(self, elem, threshold=None):
        if not threshold:
            threshold = self.threshold
        distances = self._find_distances(elem)
        knn = distances[:self.k]
        class0 = filter(lambda x: x["c"] == 0, knn)
        class1 = filter(lambda x: x["c"] == 1, knn)
        p_0 = float(len(class0)) / (len(class0) + len(class1))
        return int(p_0 < threshold), p_0

    def _find_distances(self, point):
        d = []
        point_filtered = [point[self.learn_set.get_index(x)] for x in self.columns]
        for e in self.learn_set:
            e_filtered = [e[self.learn_set.get_index(x)] for x in self.columns]
            d.append({"d": self.distance(e_filtered, point_filtered), "e": e, "c": e[self.class_column_index]})
        d.sort(key=lambda x: x["d"])
        return d

