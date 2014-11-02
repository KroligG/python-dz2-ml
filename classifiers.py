from math import sqrt


class knn_classifier:
    def __init__(self, k, distance=_euclid):
        self.k = k
        self.distance = distance

    def learn(self, learn_set, columns, class_column):
        self.learn_set = learn_set.get_rows()
        self.class_column = class_column

    def test(self, test_set):
        pass

    def classify(self, elem, treshold=0.5):
        distances = self._find_distances(elem)
        knn = distances[:self.k]
        class0 = filter(lambda x: x["e"][self.class_column] == 0, knn)
        class1 = filter(lambda x: x["e"][self.class_column] == 1, knn)
        p_0 = float(len(class0)) / (len(class0) + len(class1))
        return int(p_0 > treshold), p_0

    def _find_distances(self, point):
        d = []
        for e in self.learn_set:
            d.append({"d": self.distance(e, point), "e": e})
        d.sort(key=lambda x: x["d"])
        return d


def _euclid(x, y):
    s = map(lambda a, b: (a - b) ** 2, x, y)
    return sqrt(sum(s))
