from DeepFried2.zoo.download import download as _download

import numpy as _np
from tarfile import open as _taropen
try:  # Py2 compatibility
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def data():
    fname = _download('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
    with _taropen(fname, 'r') as f:
        with f.extractfile('cifar-100-python/train') as train:
            train = pickle.load(train, encoding='latin1')
        Xtr = _np.array(train['data'], dtype=_np.float32)
        ytr_c = _np.array(train['coarse_labels'])
        ytr_f = _np.array(train['fine_labels'])
        Xtr /= 255

        # There is no "official" validation set here that I know of!

        with f.extractfile('cifar-100-python/test') as test:
            test = pickle.load(test, encoding='latin1')
        Xte = _np.array(test['data'], dtype=_np.float32)
        yte_c = _np.array(test['coarse_labels'])
        yte_f = _np.array(test['fine_labels'])
        Xte /= 255

        # Get the label names additionally.
        with f.extractfile('cifar-100-python/meta') as m:
            m = pickle.load(m, encoding='latin1')

        try:
            from sklearn.preprocessing import LabelEncoder
            le_c = LabelEncoder()
            le_c.classes_ = _np.array(m['coarse_label_names'])
            le_f = LabelEncoder()
            le_f.classes_ = _np.array(m['fine_label_names'])
        except ImportError:
            le_c = _np.array(m['coarse_label_names'])
            le_f = _np.array(m['fine_label_names'])

    return (Xtr, ytr_c, ytr_f), (Xte, yte_c, yte_f), (le_c, le_f)


