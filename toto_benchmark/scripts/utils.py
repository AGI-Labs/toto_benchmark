import collections

class Namespace(collections.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self): 
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __repr__(self):
        return repr(self._data)
