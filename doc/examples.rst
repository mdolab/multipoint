.. _multipoint_examples:

Examples
=========

Example 1
---------
In this example we show basic operations how to create and access specific sets or processor groups.
To start we begin by importing the necessary modules.

.. literalinclude:: ../examples/ex1.py
    :start-after: # rst init (begin)
    :end-before: # rst init (end)

In this example we assume we have two codes or processor sets, ``codeA`` and ``codeB``.
We assume 3 instances of ``codeA`` where each instance requires 3, 2, and 1 processors.
For set ``codeB`` we assume only 1 instance that requires 4 processors.
Following the creation of the points sets the communicators are created.


.. literalinclude:: ../examples/ex1.py
    :start-after: # rst init (end)
    :end-before: # rst alloc (end)

Printing all values gives the following::

    setName=codeA, comm.rank=1, comm.size=3, setComm.rank=1, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[ True False False], ptID=0
    setName=codeA, comm.rank=2, comm.size=3, setComm.rank=2, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[ True False False], ptID=0
    setName=codeA, comm.rank=0, comm.size=2, setComm.rank=3, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[False  True False], ptID=1
    setName=codeA, comm.rank=1, comm.size=2, setComm.rank=4, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[False  True False], ptID=1
    setName=codeA, comm.rank=0, comm.size=1, setComm.rank=5, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[False False  True], ptID=2
    setName=codeB, comm.rank=0, comm.size=4, setComm.rank=0, setComm.size=4, setFlags={'codeA': False, 'codeB': True}, groupFlags=[ True], ptID=0
    setName=codeB, comm.rank=1, comm.size=4, setComm.rank=1, setComm.size=4, setFlags={'codeA': False, 'codeB': True}, groupFlags=[ True], ptID=0
    setName=codeB, comm.rank=2, comm.size=4, setComm.rank=2, setComm.size=4, setFlags={'codeA': False, 'codeB': True}, groupFlags=[ True], ptID=0
    setName=codeB, comm.rank=3, comm.size=4, setComm.rank=3, setComm.size=4, setFlags={'codeA': False, 'codeB': True}, groupFlags=[ True], ptID=0
    setName=codeA, comm.rank=0, comm.size=3, setComm.rank=0, setComm.size=6, setFlags={'codeA': True, 'codeB': False}, groupFlags=[ True False False], ptID=0


To perform operations on all processors in a set we can use the setFlags.
This is convenient if we want to perform the same operation on all processors in a given set.
Furthermore, if we want to access only a specific group in a set, the ``ptID`` can be conveniently used as shown.

.. literalinclude:: ../examples/ex1.py
    :start-after: # rst alloc (end)
    :end-before: # rst problem (end)
