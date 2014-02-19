.. _tutorial:

Tutorial
========

The goal of ``MultiPoint`` is to facilitate optimization problems that
contain may different computations, all occurring in parallel, each of
which may be parallel. ``MultiPoint`` effectively hides the required
MPI communication from the user which results in more readable, more
robust and easier to understand optimization scripts. 

For our simple example, lets assume we have two parallel codes, ``A``
and ``B`` that we want to run at the same time for an
optimization. The computations of ``A`` and ``B`` do not directly
depend on each other; that is they can be executed in an embarrassingly
parallel fashion.  Lets say we need to run 2 copies of code ``A``, and
one copy of code ``B``. The analysis path would look like::

                    Objective
                    Functions
                 /------------\ Funcs
             /---|   Code A   |--->----\             User supplied objcon 
             |   \------------/        |               /---------------\
  Optimizer  |   /------------\ Funcs  |  Combine      | Combine funcs |  Return to 
  ----->---- +---|   Code A   |--->--- +------>--------+ to get final  |----->------
  Input      |   \------------/        |  all funcs    | obj/con       |  optimizer
             |   /------------\ Funcs  |               \---------------/
             \---|   Code B   |--->----/
                 \------------/


Lets also assume, that the first copy of code ``A`` requires 3
processors, the second copy of code ``A`` requires 2 processors and
the copy of code ``B`` requires 4 processors. For this case, we would
require ``3 + 2 + 4 = 7`` total processors. Scripts using
``MultiPointSparse`` must be called with precisely the correct
number of processors. 

  >>> from multipoint import * 
  >>> MP = multiPointSparse(MPI.COMM_WORLD)
  >>> MP.addProcessorSet('codeA', 2, [3, 2])
  >>> MP.addProcessorSet('codeB', 1, 4)

The input to each of the Objective Functions is the (unmodified) dictionary of
optimization variables from pyOptSparse. Each code is then required to
use the optimization variables as it requires. 

The output from each of Objective functions ``funcs`` is a Python
dictionary of computed values. **For computed values that are
different for each member in a processorSet or between processorSets
it is necessary to use unique keys**.  It is therefore necessary for
the user to use an appropriate name mangling scheme. 

In the example above we have two copies of Code A. In typical usage,
these two instances will produce the same *number* and *type* of
quantities but at different operating conditions or other similar
variation. Since we need these quantities for either the optimization
objective or constraints, these values must be given a unique name. 

A simple name-mangling scheme is to simply use the ``ptID`` variable that
is returned from the call to `createCommunicators`::

  def objA(x):
      funcs['A_%d'%ptID] = function_of_x()

      return funcs

A similar thing can be done for ``B``::

  def objB(x):
      funcs['B_%d'%ptID] = function_of_x()

      return funcs

A ``processorSet`` is characterized by a single "objective" and
"sensitivity" function. For each ``processorSet`` we must supply Python
functions for the objective and sensitivity evaluation. 

    >>> MP.setProcSetObjFunc('codeA', objA)
    >>> MP.setProcSetObjFunc('codeB', objB)
    >>> MP.setProcSetSensFunc('codeA', sensA)
    >>> MP.setProcSetSensFunc('cdoeB', sensB)

The functions ``sensA`` and ``sensB`` must compute derivatives of the
functionals with respect to the design variables defined in the
``optProb`` Optimization problem class. Derivatives use the dictionary
sensitivity return format described in ``pyOptSparse`` documentation.

``multiPointSparse`` will then automatically communicate the values
and call the user supplied ``objcon`` function with the total set of
functions. The purpose of ``objcon`` is to combine functions from the
individual objective functions to form the final objective and
constraint dictionary for ``pyOptSparse``. A schematic of this
process is given below::

                       Pass-through keys
                 /----------->--------------------\
    all funcs    |                                |   output to pyOptSparse
  ------->------ + input     /--------\ output    |------------->----
                 \-------->--+ objcon | ----------/
                   keys      \--------/ keys 

``multiPointSparse`` analyzes the optimization object and determine if
any of the required constraint keys are already present in all funcs,
these keys are flagged as "pass-through"...that is they "by-pass"
entirely the ``objcon`` function. The purpose therefore of objcon is
to use the remaining functions in ``all funcs`` (the ``input keys``)
to compute the remainder of the required constraints (``output keys``)
and objective. For example::

  def objcon(funcs):
     fobj = 0.0
     for i in range(2):
         fobj += funcs['A_%d'%i]

     fobj /= funcs[B_0]

     fcon['B_con'] = funcs[B_0]/funcs[A_0]

     return fobj, fcon

There all three values contribute to the objective, while ``A_0`` and
``B_0`` combine to form the constraint ``B_con``. This example has no
``pass-though keys``.

 Generally speaking, the computations in objcon should be simple and
not overally computationally intensive. The sensitivity of the ``output
keys`` with respect to the ``input keys`` is computed automatically by
``multiPointSparse`` using the complex step method.

.. warning::
   Pass-through keys **cannot** be used in objcon. 

.. warning:: 
  Computations in objcon must be able to use complex
  number. Generally this will mean if numpy arrays are used, the
  ``dtype=complex`` keyword argument is used.


The ``objcon`` function is set using the call::

    >>> MP.setObjCon(objCon)

As noted earlier, ``multiPointSparse`` uses the optimization problem
to determine which keys are already constraints and which need to be
combined in ``objcon``.  This is done using::

    >>> optProb = Optimization('opt', MP.obj)
    >>> # Setup optimization problem
    >>> # MP needs the optProb after everything is setup.
    >>> MP.setOptProb(optProb)
    >>> # Create optimizer and use MP.sens for the sensitivity function on opt call
    >>> snopt(optProb, sens=MP.sens, ...)
