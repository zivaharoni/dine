class VanillaIterator(object):
    def __init__(self, ds):
        self.data = ds


    def __iter__(self):
        self.iter = iter(self.data)
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except:
            raise StopIteration

    def __len__(self):
        return len(list(self.data.as_numpy_iterator()))

class FixedLengthIterator(object):
    def __init__(self, ds, length=None):
        self.data = ds
        self.len = length
        self.count = 0

    def __iter__(self):
        self.iter = iter(self.data)
        self.count = 0
        return self

    def __next__(self):
        try:
            if self.len is None or self.count < self.len:
                self.count += 1
                return next(self.iter)
            else:
                raise StopIteration
        except:
            raise StopIteration

    def __len__(self):
        return len(list(self.data.as_numpy_iterator()))
