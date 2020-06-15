__version__ = '1.2.0'


from .multiPoint import multiPoint
from .multiPointSparse import multiPointSparse
from .multiPointSparse import createGroups
from .multiPointSparse import redirectIO

__all__ = ['multiPoint', 'multiPointSparse', 'createGroups', 'redirectIO']
