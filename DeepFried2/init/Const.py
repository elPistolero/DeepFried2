import numpy as _np


def const(value):
    def init(shape, fan):
        return _np.full(shape, value)
    return init


def array(value):
    def init(shape, fan):
        a = _np.array(value, copy=True)
        assert a.shape == shape, "Shape mismatch in initializer: provided {}, requested {}".format(a.shape, shape)
        return a
    return init
