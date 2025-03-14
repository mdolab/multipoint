# =============================================================================
# Imports
# =============================================================================
import os
import inspect
import types
import copy
from collections import OrderedDict

import numpy as np
from mpi4py import MPI

from baseclasses.utils import Error
from .utils import dkeys, skeys, _extractKeys, _complexifyFuncs


# =============================================================================
# MultiPoint Class
# =============================================================================
class multiPointSparse:
    """
    Create the multiPoint class on the provided comm.

    Parameters
    ----------
    gcomm : MPI.Intracomm
        Global MPI communicator from which all processor groups
        are created. It is usually MPI_COMM_WORLD but may be
        another intraCommunicator that has already been created.

    Examples
    --------
    We will setup a multipoint problem with two procSets: a 'cruise'
    set with 3 members and 32 procs each, and a maneuver set with two
    members with 10 and 20 procs respectively. Our script will have to
    define 5 python functions:

    #. Evaluate functions for cruise::

         def cruiseObj(x):
             funcs = {} # Fill up with functions
             ...
             return funcs

    #. Evaluate functions for maneuver::

         def maneuverObj(x):
             funcs = {} # Fill up with functions
             ...
             return funcs

    #. Evaluate function sensitivity for cruise::

         def cruiseSens(x, funcs):
             funcSens = {}
             ...
             return funcSens

    #. Evaluate function sensitivity for cruise::

        def maneuverSens(x, funcs):
             funcSens = {}
             ...
             return funcSens

    #. Function to compute addition functions::


        def objCon(funcs):
             funcs['new_func'] = combination_of_funcs
             ...
             return funcs

    >>> MP = multiPointSparse.multiPoint(MPI.COMM_WORLD)
    >>> MP.addProcessorSet('cruise', 3, 32)
    >>> MP.addProcessorSet('maneuver', 2, [10, 20])
    >>> # Possibly create directories
    >>> ptDirs = MP.createDirectories('/home/user/output/')
    >>> # Get the communicators and flags
    >>> comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
    >>> # Setup problems and python functions
    >>> ....
    >>> MP.setProcSetObjFunc('cruise', cruiseObj)
    >>> MP.setProcSetObjFunc('maneuver', maneuverObj)
    >>> MP.setProcSetSensFunc('cruise', cruiseSens)
    >>> MP.setProcSetSensFunc('maneuver', maneuverSens)
    >>> MP.setObjCon(objCon)
    >>> # Create optimization problem using MP.obj
    >>> optProb = Optimization('opt', MP.obj)
    >>> # Setup optimization problem
    >>> # MP needs the optProb after everything is setup.
    >>> MP.setOptProb(optProb)
    >>> # Create optimizer and use MP.sens for the sensitivity function on opt call
    >>> snopt(optProb, sens=MP.sens, ...)

    Notes
    -----
    multiPointSparse requires ``useGroups=True`` (default) when creating
    the optProb (Optimization instance).
    """

    def __init__(self, gcomm):
        assert isinstance(gcomm, MPI.Intracomm)
        self.gcomm = gcomm
        self.pSet = OrderedDict()
        self.dummyPSet = set()
        self.pSetRoot = None
        self.objective = None
        self.setFlags = None
        self.constraints = None
        self.cumSets = [0]
        self.objCommPattern = None
        self.sensCommPattern = None
        # User-specified function
        self.userObjCon = None
        self.nUserObjConArgs = None

        # Information used for determining keys for CS loop
        self.conKeys = set()
        self.outputWRT = {}
        self.outputSize = {}
        self.dvSize = {}
        self.dvsAsFuncs = []
        self.consAsInputs = []
        self.funcs = None
        self.inputKeys = None
        self.outputKeys = None
        self.passThroughKeys = None

    def addProcessorSet(self, setName, nMembers, memberSizes):
        """
        A Processor set is defined as one or more groups of processors
        that use the same obj() and sens() routines. Members of
        processor sets typically, but not necessarily, return the same
        number of functions. In all cases, the function names must be
        unique.

        Parameters
        ----------
        setName : str
            Name of process set. Process set names must be unique.

        nMembers : int
            Number of members in the set.

        memberSizes : int, iteratable
            Number of processors on each set. If an integer is supplied all
            members use the same number of processors.
            If a list or array is provided, a different number of processors
            on each member can be specified.

        Examples
        --------
        >>> MP = multiPointSparse.multiPoint(MPI.COMM_WORLD)
        >>> MP.addProcessorSet('cruise', 3, 32)
        >>> MP.addProcessorSet('maneuver', 2, [10, 20])

        The ``cruise`` set creates 3 processor groups, each of size 32.
        and the ``maneuver`` set creates 2 processor groups, of size 10 and 20.
        """
        # Lets let the user explicitly set nMembers to 0. This is
        # equivalent to just turning off that proc set.
        if nMembers == 0:
            self.dummyPSet.add(setName)
        else:
            nMembers = int(nMembers)
            memberSizes = np.atleast_1d(memberSizes)
            if len(memberSizes) == 1:
                memberSizes = np.ones(nMembers) * memberSizes[0]
            else:
                if len(memberSizes) != nMembers:
                    raise Error("The supplied memberSizes list is not the correct length.")

            self.pSet[setName] = procSet(setName, nMembers, memberSizes, len(self.pSet))

    def createCommunicators(self):
        """
        Create the communicators after all the procSets have been
        added. All procSets MUST be added before this routine is
        called.

        Returns
        -------
        comm : MPI.Intracomm
            This is the communicator for the member of the procSet. Basically,
            this is the communicator that the (parallel) analysis should be
            created on.
        setComm : MPI.Intracomm
            This is the communicator that spans the entire processor set.
        setFlags : dict
            This is a dictionary whose entry for ``setName``, as specified in
            addProcessorSet() is True on a processor belonging to that set.
        groupFlags : list
            This list is used to distinguish between members within
            a processor set. This list of of length nMembers and the
            ith entry is true for the ith group.
        ptID : int
            This is the index of the group that this processor belongs to.

        Examples
        --------
        >>> MP = multiPointSparse.multiPoint(MPI.COMM_WORLD)
        >>> MP.addProcessorSet('cruise', 3, 32)
        >>> MP.addProcessorSet('maneuver', 2, [10, 20])
        >>> comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

        The following will be true for all processors for the second member
        of the ``cruise`` procSet.

        >>> setFlags['cruise'] and groupFlags[1] == True
        """

        # First we determine the total number of required procs:
        nProc = 0
        for setName in dkeys(self.pSet):
            nProc += self.pSet[setName].nProc

        # Check the sizes
        if nProc < self.gcomm.size or nProc > self.gcomm.size:
            raise Error(f"multiPointSparse must be called with EXACTLY {nProc} processors.")

        # Create a cumulative size array
        setCount = len(self.pSet)
        setSizes = np.zeros(setCount)
        for setName in dkeys(self.pSet):
            setSizes[self.pSet[setName].setID] = self.pSet[setName].nProc

        cumSets = np.zeros(setCount + 1, "intc")
        for i in range(setCount):
            cumSets[i + 1] = cumSets[i] + setSizes[i]

        setFlags = {}

        # Determine the member_key for each processor
        memberKey = None
        for key in dkeys(self.pSet):
            if self.gcomm.rank >= cumSets[self.pSet[key].setID] and self.gcomm.rank < cumSets[self.pSet[key].setID + 1]:
                memberKey = self.pSet[key].setID
                setFlags[self.pSet[key].setName] = True
            else:
                setFlags[self.pSet[key].setName] = False

        setComm = self.gcomm.Split(memberKey)

        # Set this new_comm into each pSet and let each procSet create
        # its own split:
        comm = None
        groupFlags = None
        ptID = None
        for key in dkeys(self.pSet):
            if setFlags[key]:
                self.pSet[key].gcomm = setComm
                self.pSet[key].createCommunicators()

                self.gcomm.barrier()

                comm = self.pSet[key].comm
                groupFlags = self.pSet[key].groupFlags
                ptID = self.pSet[key].groupID

        self.setFlags = setFlags
        # Now just append the dummy procSets:
        for key in skeys(self.dummyPSet):
            self.setFlags[key] = False

        self.pSetRoot = {}
        for key in dkeys(self.pSet):
            self.pSetRoot[key] = cumSets[self.pSet[key].setID]

        return comm, setComm, setFlags, groupFlags, ptID

    def getSetName(self):
        """After MP.createCommunicators is call, this routine may be called
        to return the name of the set that this processor belongs
        to. This may result in slightly cleaner script code.

        Returns
        -------
        setName : str
            The name of the set that this processor belongs to.
        """

        for iset in dkeys(self.setFlags):
            if self.setFlags[iset]:
                return iset

    def createDirectories(self, rootDir):
        """
        This function can be called only after all the procSets have
        been added. This can facilitate distinguishing output files
        when there are a large number of procSets and/or members of
        procSets.

        Parameters
        ----------
        rootDir : str
            Root path where directories are to be created

        Returns
        -------
        ptDirs : dict
            A dictionary of all the created directories. Each dictionary
            entry has key defined by 'setName' and contains a list of size
            nMembers, each entry of which is the path to the created
            directory

        Examples
        --------
        >>> MP = multiPointSparse.multiPoint(MPI.COMM_WORLD)
        >>> MP.addProcessorSet('cruise', 3, 32)
        >>> MP.addProcessorSet('maneuver', 2, [10, 20])
        >>> ptDirs = MP.createDirectories('/home/user/output/')
        >>> ptDirs
        {'cruise': ['/home/user/output/cruise_0',
                    '/home/user/output/cruise_1',
                    '/home/user/output/cruise_2'],
         'maneuver': ['/home/user/output/maneuver_0',
                      '/home/user/output/maneuver_1']}
        """

        if len(self.pSet) == 0:
            return

        ptDirs = {}
        for key in dkeys(self.pSet):
            ptDirs[key] = []
            for i in range(self.pSet[key].nMembers):
                dirName = os.path.join(rootDir, f"{self.pSet[key].setName}_{i}")
                ptDirs[key].append(dirName)

                if self.gcomm.rank == 0:  # Only global root proc makes
                    # directories
                    os.system(f"mkdir -p {dirName}")

        return ptDirs

    def setProcSetObjFunc(self, setName, func):
        """
        Set a single python function handle to compute the functionals

        Parameters
        ----------
        setName : str
            Name of set we are setting the function for
        func : Python function
            Python function handle
        """
        if setName in self.dummyPSet:
            return
        if setName not in self.pSet:
            raise Error(f"setName '{setName}' has not been added with addProcessorSet.")
        if not isinstance(func, types.FunctionType):
            raise Error("func must be a Python function handle.")

        self.pSet[setName].objFunc = [func]

    def setProcSetSensFunc(self, setName, func):
        """
        Set the python function handle to compute the derivative of
        the functionals

        Parameters
        ----------
        setName : str
            Name of set we are setting the function for
        func : Python function
            Python function handle
        """
        if setName in self.dummyPSet:
            return
        if setName not in self.pSet:
            raise Error(f"setName '{setName}' has not been added with addProcessorSet.")
        if not isinstance(func, types.FunctionType):
            raise Error("func must be a Python function handle.")

        self.pSet[setName].sensFunc = [func]

    def addProcSetObjFunc(self, setName, func):
        """
        Add an additional python function handle to compute the functionals

        Parameters
        ----------
        setName : str
            Name of set we are setting the function for
        func : Python function
            Python function handle
        """
        if setName in self.dummyPSet:
            return
        if setName not in self.pSet:
            raise Error(f"setName '{setName}' has not been added with addProcessorSet.")
        if not isinstance(func, types.FunctionType):
            raise Error("func must be a Python function handle.")

        self.pSet[setName].objFunc.append(func)

    def addProcSetSensFunc(self, setName, func):
        """
        Add an additional python function handle to compute the
        derivative of the functionals

        Parameters
        ----------
        setName : str
            Name of set we are setting the function for
        func : Python function
            Python function handle

        """
        if setName in self.dummyPSet:
            return

        if setName not in self.pSet:
            raise Error(f"setName '{setName}' has not been added with addProcessorSet.")
        if not isinstance(func, types.FunctionType):
            raise Error("func must be a Python function handle.")

        self.pSet[setName].sensFunc.append(func)

    def setObjCon(self, func):
        """
        Set the python function handle to compute the final objective
        and constraints that are combinations of the functionals.

        Parameters
        ----------
        func : Python function
            Python function handle
        """
        if not isinstance(func, types.FunctionType):
            raise Error("func must be a Python function handle.")

        # Also do some checking on function prototype to make sure it is ok:
        sig = inspect.signature(func)
        if len(sig.parameters) not in [1, 2, 3]:
            raise Error(
                "The function signature for the function given to 'setObjCon' is invalid. It must be: "
                + "def objCon(funcs):, def objCon(funcs, printOK): or def objCon(funcs, printOK, passThroughFuncs):"
            )

        # Now we know that there are exactly one or two arguments.
        self.nUserObjConArgs = len(sig.parameters)
        self.userObjCon = func

    def setOptProb(self, optProb):
        """
        Set the optimization problem that this multiPoint object will
        be used for. This is required for this class to know how to
        assemble the gradients. If the optProb is not 'finished', it
        will done so here. Therefore, this function is collective on
        the comm that optProb is built on. multiPoint sparse does
        *not* hold a reference to optProb so no additional changes can
        be made to optProb after this function is called.

        Parameters
        ----------
        optProb : pyOptSparse optimization problem class
            The optProb object to use
        """
        optProb.finalize()

        # Since there is no distinction between objective(s) and
        # constraints just put everything in conKeys, including the
        # objective(s)
        for iCon in dkeys(optProb.constraints):
            if not optProb.constraints[iCon].linear:
                self.conKeys.add(iCon)
                self.outputWRT[iCon] = optProb.constraints[iCon].wrt
                self.outputSize[iCon] = optProb.constraints[iCon].ncon
        for iObj in dkeys(optProb.objectives):
            self.conKeys.add(iObj)
            self.outputWRT[iObj] = list(optProb.variables.keys())
            self.outputSize[iObj] = 1

        for dvGroup in dkeys(optProb.variables):
            ss = optProb.dvOffset[dvGroup]
            self.dvSize[dvGroup] = ss[1] - ss[0]

        self.conKeys = set(self.conKeys)

        # Check the dvsAsFuncs names to make sure they are *actually*
        # design variables and raise error
        for dv in self.dvsAsFuncs:
            if dv not in optProb.variables:
                raise Error(
                    (
                        "The supplied design variable '{}' in addDVsAsFunctions() call"
                        + " does not exist in the supplied Optimization object."
                    ).format(dv)
                )

    def addDVsAsFunctions(self, dvs):
        """This function allows you to specify a list of design variables to
        be explicitly used as functions. Essentially, we just copy the
        values of the DVs directly into keys in 'funcs' and
        automatically generate an identity Jacobian. This allows the
        remainder of the objective/sensitivity computations to be
        proceed as per usual.

        Parameters
        ----------
        dvs : string or list of strings
           The DV names the user wants to use directly as functions
        """

        if isinstance(dvs, str):
            self.dvsAsFuncs.append(dvs)
        elif isinstance(dvs, list):
            self.dvsAsFuncs.extend(dvs)

    def addConsAsObjConInputs(self, cons):
        """
        This function allows functions to be used both as constraints,
        as well as inputs to the ObjCon, therefore no longer bypassed.

        Parameters
        ----------
        cons : string or list of strings
           The constraint names the user wants to use as ObjCon inputs
        """

        if isinstance(cons, str):
            self.consAsInputs.append(cons)
        elif isinstance(cons, list):
            self.consAsInputs.extend(cons)

    def obj(self, x):
        """
        This is a built-in objective function that is designed to be
        used directly as an objective function with pyOptSparse. The
        user should not use this function directly, instead see the
        class documentation for the intended usage.

        Parameters
        ----------
        x : dict
            Dictionary of variables returned from pyOptSparse
        """
        for key in dkeys(self.pSet):
            if self.setFlags[key]:
                # Run "obj" function to generate functionals
                res = {"fail": False}
                for func in self.pSet[key].objFunc:
                    tmp = func(x)
                    if tmp is None:
                        raise Error(
                            (
                                "No return from user supplied objective function for pSet {}. "
                                + "Functional derivatives must be returned in a dictionary."
                            ).format(key)
                        )

                    if "fail" in tmp:
                        res["fail"] = bool(tmp.pop("fail") or res["fail"])
                    res.update(tmp)

        if self.objCommPattern is None:
            # On the first pass we need to determine the (one-time)
            # communication pattern

            # Send all the keys
            allKeys = self.gcomm.allgather(sorted(res.keys()))

            self.objCommPattern = {}

            for i in range(len(allKeys)):  # This is looping over processors
                for key in allKeys[i]:  # This loops over keys from proc
                    if key not in self.objCommPattern:
                        if key != "fail":
                            # Only add on the lowest proc and ignore on higher
                            # ones
                            self.objCommPattern[key] = i

        # Perform Communication of functionals
        allFuncs = {}
        for key in dkeys(self.objCommPattern):
            if self.objCommPattern[key] == self.gcomm.rank:
                tmp = self.gcomm.bcast(res[key], root=self.objCommPattern[key])
            else:
                tmp = self.gcomm.bcast(None, root=self.objCommPattern[key])

            allFuncs[key] = tmp

        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res["fail"], op=MPI.LOR)

        # Add in the extra DVs as Funcs...can do this on all procs
        # since all procs have the same x
        for dv in self.dvsAsFuncs:
            allFuncs[dv] = x[dv]

        # Save the functions since we need these for the derivatives
        self.funcs = copy.deepcopy(allFuncs)

        # Determine which additional keys are necessary:
        funckeys = set(allFuncs.keys())
        # Input Keys are the input variables to the objCon function
        # Output Keys are the output variables from the objCon function
        self.inputKeys = funckeys.difference(self.conKeys)  # input = func - con
        self.outputKeys = self.conKeys.difference(funckeys)  # output = con - func
        self.passThroughKeys = funckeys.intersection(self.conKeys)  # passThrough = func & con

        # Manage any keys that are both inputs and constraints (consAsInputs)
        # Check consAsFuncs only contains keys contained in passThoughKeys
        # inputKeys += consAsInputs
        # passThroughKeys -= consAsInputs
        if len(self.consAsInputs) > 0:
            self.consAsInputs = set(self.consAsInputs)
            self.consAsInputs.intersection_update(self.passThroughKeys)
            self.inputKeys.update(self.consAsInputs)
            self.passThroughKeys.difference_update(self.consAsInputs)

        inputFuncs = _extractKeys(allFuncs, self.inputKeys)
        passThroughFuncs = _extractKeys(allFuncs, self.passThroughKeys)
        funcs = self._userObjConWrap(inputFuncs, True, passThroughFuncs)

        # Add the pass-through ones back:
        funcs.update(passThroughFuncs)

        (funcs, fail) = self.gcomm.bcast((funcs, fail), root=0)

        return funcs, fail

    def sens(self, x, funcs):
        """
        This is a built-in sensitivity function that is designed to be
        used directly as a the sensitivity function with
        pyOptSparse. The user should not use this function directly,
        instead see the class documentation for the intended usage.

        Parameters
        ----------
        x : dict
            Dictionary of variables returned from pyOptSparse
        """
        for key in dkeys(self.pSet):
            if self.setFlags[key]:
                # Run "sens" function to functionals sensitivities
                res = {"fail": False}
                for func in self.pSet[key].sensFunc:
                    tmp = func(x, funcs)
                    if tmp is None:
                        raise Error(
                            (
                                "No return from user supplied sensitivity function for pSet {}. "
                                + "Functional derivatives must be returned in a dictionary."
                            ).format(key)
                        )
                    if "fail" in tmp:
                        res["fail"] = bool(tmp.pop("fail") or res["fail"])

                    res.update(tmp)

        if self.sensCommPattern is None:
            # On the first pass we need to determine the (one-time)
            # communication pattern

            # Send all the keys
            allKeys = self.gcomm.allgather(sorted(res.keys()))

            self.sensCommPattern = {}

            for i in range(len(allKeys)):  # This is looping over processors
                for key in allKeys[i]:  # This loops over keys from proc
                    if key not in self.sensCommPattern:
                        if key != "fail":
                            # Only add on the lowest proc and ignore on higher ones
                            self.sensCommPattern[key] = i

        # Perform Communication of functional (derivatives)
        funcSens = {}
        for key in dkeys(self.sensCommPattern):
            if self.sensCommPattern[key] == self.gcomm.rank:
                tmp = self.gcomm.bcast(res[key], root=self.sensCommPattern[key])
            else:
                tmp = self.gcomm.bcast(None, root=self.sensCommPattern[key])

            funcSens[key] = tmp

        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res["fail"], op=MPI.LOR)

        # Add in the sensitivity of the extra DVs as Funcs...This will
        # just be an identity matrix
        for dv in self.dvsAsFuncs:
            if np.isscalar(x[dv]) or len(np.atleast_1d(x[dv])) == 1:
                funcSens[dv] = {dv: np.eye(1)}
            else:
                funcSens[dv] = {dv: np.eye(len(x[dv]))}

        # Now we have to perform the CS loop over the user-supplied
        # objCon function to generate the derivatives of our final
        # constraints (and objective(s)) with respect to the
        # intermediate functionals. We will put everything in gcon
        # (including the objective)

        gcon = {}
        # Extract/Complexify just the keys we need:
        passThroughFuncs = _extractKeys(self.funcs, self.passThroughKeys)
        cFuncs = _extractKeys(self.funcs, self.inputKeys)
        cFuncs = _complexifyFuncs(cFuncs, self.inputKeys)

        # Just copy the passthrough keys and keys that are both inputs and constrains:
        for pKey in self.passThroughKeys:
            gcon[pKey] = funcSens[pKey]
        for cKey in self.consAsInputs:
            gcon[cKey] = funcSens[cKey]

        # Setup zeros for the output keys:
        for oKey in skeys(self.outputKeys):
            gcon[oKey] = {}
            # Only loop over the DVsets that this constraint has:
            for dvSet in self.outputWRT[oKey]:
                gcon[oKey][dvSet] = np.zeros((self.outputSize[oKey], self.dvSize[dvSet]))

        for iKey in skeys(self.inputKeys):  # Keys to peturb:
            if np.isscalar(cFuncs[iKey]) or len(np.atleast_1d(cFuncs[iKey])) == 1:
                cFuncs[iKey] += 1e-40j
                con = self._userObjConWrap(cFuncs, False, passThroughFuncs)
                cFuncs[iKey] -= 1e-40j

                # Extract the derivative of output key variables
                for oKey in skeys(self.outputKeys):
                    n = self.outputSize[oKey]
                    for dvSet in self.outputWRT[oKey]:
                        if dvSet in funcSens[iKey]:
                            deriv = (np.imag(np.atleast_1d(con[oKey])) / 1e-40).reshape((n, 1))
                            gcon[oKey][dvSet] += np.dot(deriv, np.atleast_2d(funcSens[iKey][dvSet]))

            else:
                for i in range(len(cFuncs[iKey])):
                    cFuncs[iKey][i] += 1e-40j
                    con = self._userObjConWrap(cFuncs, False, passThroughFuncs)
                    cFuncs[iKey][i] -= 1e-40j

                    # Extract the derivative of output key variables
                    for oKey in skeys(self.outputKeys):
                        n = self.outputSize[oKey]

                        for dvSet in self.outputWRT[oKey]:
                            if dvSet in funcSens[iKey]:
                                deriv = (np.imag(np.atleast_1d(con[oKey])) / 1e-40).reshape((n, 1))
                                gcon[oKey][dvSet] += np.dot(deriv, np.atleast_2d(funcSens[iKey][dvSet][i, :]))

        gcon = self.gcomm.bcast(gcon, root=0)
        fail = self.gcomm.bcast(fail, root=0)

        return gcon, fail

    def _userObjConWrap(self, funcs, printOK, passThroughFuncs):
        """Small wrapper to determine how to call user function:"""
        if self.nUserObjConArgs == 1:
            return self.userObjCon(funcs)
        elif self.nUserObjConArgs == 2:
            if self.gcomm.rank == 0:
                return self.userObjCon(funcs, printOK)
            else:
                return self.userObjCon(funcs, False)
        elif self.nUserObjConArgs == 3:
            if self.gcomm.rank == 0:
                return self.userObjCon(funcs, printOK, passThroughFuncs)
            else:
                return self.userObjCon(funcs, False, passThroughFuncs)


class procSet:
    """
    A container class to bundle information pertaining to a specific
    processor set. It is not intended to be used externally by a user.
    No error checking is performed since the multiPoint class should
    have already checked the inputs.
    """

    def __init__(self, setName, nMembers, memberSizes, setID):
        self.setName = setName
        self.nMembers = nMembers
        self.memberSizes = memberSizes
        self.nProc = np.sum(self.memberSizes)
        self.gcomm = None
        self.objFunc = []
        self.sensFunc = []
        self.cumGroups = None
        self.groupID = None
        self.groupFlags = None
        self.comm = None
        self.setID = setID

    def createCommunicators(self):
        """
        Once the comm for the procSet is determined, we can split up
        this comm as well
        """
        # Create a cumulative size array
        cumGroups = np.zeros(self.nMembers + 1, "intc")

        for i in range(self.nMembers):
            cumGroups[i + 1] = cumGroups[i] + self.memberSizes[i]

        # Determine the member_key (m_key) for each processor
        m_key = None
        for i in range(self.nMembers):
            if self.gcomm.rank >= cumGroups[i] and self.gcomm.rank < cumGroups[i + 1]:
                m_key = i

        self.comm = self.gcomm.Split(m_key)
        self.groupFlags = np.zeros(self.nMembers, bool)
        self.groupFlags[m_key] = True
        self.groupID = m_key
        self.cumGroups = cumGroups
