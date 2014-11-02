from itertools import izip
from math import sqrt
from random import shuffle as random_shuffle


class ml_table:
    def __init__(self, rows=None, headers=False):
        self._headers = []
        self._table = []
        self._rowlen = -1
        self._types = []
        if headers:
            self.set_headers(headers)
        if rows:
            self.set_rows(rows)

    def set_headers(self, headers):
        if self._rowlen == -1:
            self._rowlen = len(headers)
        if len(headers) != self._rowlen:
            raise Exception(' '.join(["Headers length is", len(headers), "but expected length is", self._rowlen]))
        self._headers = headers[:]

    def set_rows(self, rows):
        if rows:
            for row in rows:
                self.add_row(row, enforce_types=False)
            self.__enforce_types()

    def add_row(self, row, enforce_types=True):
        if self._rowlen == -1:
            self._rowlen = len(row)
            if not self._headers:
                self._headers = [""] * self._rowlen
        if len(row) != self._rowlen:
            raise Exception(' '.join(["Row length is", len(row), "but expected length is", self._rowlen]))
        self.__update_types(row)
        self._table.append(row[:])
        if enforce_types:
            self.__enforce_types()

    def get_rows(self):
        return self._table[:]

    def get_headers(self):
        return self._headers[:]

    def __update_types(self, row):
        next_type = {int: float, float: str}
        if not self._types:
            self._types = [int] * len(row)
        for i, val in enumerate(row):
            if isinstance(val, (int, float)):
                self._types[i] = type(val)
            elif isinstance(val, str):
                while not _is_type(val, self._types[i]):
                    self._types[i] = next_type[self._types[i]]
            else:
                raise ValueError("Supported types are: int, float, str")

    def __enforce_types(self):
        for row in self._table:
            for i in xrange(len(row)):
                row[i] = self._types[i](row[i])

    @classmethod
    def load_table(cls, filename, has_headers=False, separator=",", shuffle=False):
        table = cls()
        rows = [[val.strip() for val in line.split(separator)] for line in open(filename, 'r')]
        if has_headers:
            table.set_headers(rows[0])
            rows = rows[1:]
        if shuffle:
            random_shuffle(rows)
        table.set_rows(rows)
        return table

    def shuffle(self):
        random_shuffle(self._table)

    def print_table(self, write_headers=True, separator=","):
        result = []
        if write_headers and self._headers:
            result.append(separator.join(self._headers))
        for row in self._table:
            result.append(separator.join((str(v) for v in row)))
        return "\n".join(result)

    def save_table(self, filename, write_headers=True, separator=","):
        with open(filename, 'w') as f:
            f.write(self.print_table(write_headers=write_headers, separator=separator))

    def get_header(self, index):
        return self._headers[index]

    def set_header(self, index, value):
        self._headers[index] = value

    def get_column(self, key):
        index = self.get_index(key)
        return [row[index] for row in self._table]

    def get_index(self, key):
        if isinstance(key, (int, slice)):
            return key
        else:
            return self._headers.index(key)

    def get_columns(self):
        return list(_transpose(self._table))

    def insert_columns(self, columns, index=None, headers=None, replace=False):
        table_columns = self.get_columns()
        if not index:
            index = len(table_columns)
        for i, c in enumerate(columns):
            column = c[:]
            t = _update_type(column)
            if replace:
                table_columns[index] = column
                self._types[index] = t
                if headers:
                    self._headers[index] = headers[i]
                replace = False
            else:
                table_columns.insert(index + i, column)
                self._types.insert(index + i, t)
                self._headers.insert(index + i, headers[i] if headers else "")
        self._table = list(_transpose(table_columns))

    def insert_column(self, column, index=None, header=None, replace=False):
        self.insert_columns([column], index=index, headers=[header] if header else None, replace=replace)

    def normalize(self, key, method="maxmin", replace=True):
        index = self.get_index(key)
        column = self.get_column(key)
        method_fun = _normalization_methods[method]
        normalized_column = method_fun(column)
        header = None if replace else self.get_header(index) + ":" + method
        self.insert_column(normalized_column, index=index if replace else index + 1, header=header, replace=replace)

    def transform_enum_to_bin(self, key, values=None, replace=False):
        index = self.get_index(key)
        column = self.get_column(index)
        column_name = self._headers[index]
        if not values:
            values = set(column)
        new_columns = [[int(c == v) for c in column] for v in values]
        headers = [':'.join((column_name, v)) for v in values]
        self.insert_columns(new_columns, index=index if replace else index + 1, headers=headers, replace=replace)


    def get_statistics(self, key, quantiles=None):
        column = self.get_column(key)
        return _get_statistics(column, quantiles=quantiles)

    def __len__(self):
        return len(self._table)

    def __getitem__(self, key):
        index = self.get_index(key)
        if isinstance(index, int):
            return self._table[index]
        elif isinstance(index, slice):
            return ml_table(headers=self._headers, rows=self._table[index])

    def __iter__(self):
        return iter(self._table)

    def __str__(self):
        return self.print_table()

    def __repr__(self):
        return self.print_table()


def _transpose(matrix):
    return (list(x) for x in izip(*matrix))


def _is_type(s, number_type):
    try:
        number_type(s)
        return True
    except ValueError:
        return False


def _get_statistics(column, quantiles=None):
    l = len(column)
    avg = float(sum(column)) / l
    d = sum(((x - avg) ** 2 for x in column)) / l
    stats = {
        "fill_count": l,
        "min": min(column),
        "max": max(column),
        "avg": avg,
        "dispersion": d
    }
    if quantiles:
        sorted_column = sorted(column)
        q_arr = [sorted_column[int(l * q)] for q in quantiles]
        stats["quantiles"] = q_arr
    return stats


def _maxmin(column):
    st = _get_statistics(column)
    max_e = st["max"]
    min_e = st["min"]
    div = (max_e - min_e)
    return [float(c - min_e) / div for c in column]


def _dispersion(column):
    st = _get_statistics(column)
    avg = st["avg"]
    std = sqrt(st["dispersion"])
    return [float(c - avg) / std for c in column]


_normalization_methods = {
    "maxmin": _maxmin,
    "dispersion": _dispersion
}

def _update_type(column):
    next_type = {int: float, float: str}
    t = int
    for val in column:
        if isinstance(val, (int, float)):
            t = type(val)
        elif isinstance(val, str):
            while not _is_type(val, t):
                t = next_type[t]
        else:
            raise ValueError("Supported types are: int, float, str")
    for i in xrange(len(column)):
        column[i] = t(column[i])
    return t