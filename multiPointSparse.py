#!/usr/bin/python
"""
multiPointSparse.py -- A python utility for aiding complicated multi-point
optimizations -- designed to work specifically with pyOptSparse. 

Copyright (c) 2013 by Dr. G. K. W. Kenway
All rights reserved. Not to be used for commercial purposes.

Developers:
-----------
- Dr. G. K. W. Kenway

History
-------
v. 1.0  - First implementation
"""
from __future__ import print_function 
# =============================================================================
# Imports
# =============================================================================
import os
import inspect
import types
import copy
import sys
try:
    from collections import OrderedDict
except ImportError:
    # python 2.6 or earlier, use backport
    # get using "pip install ordereddict"
    from ordereddict import OrderedDict
import numpy
from mpi4py import MPI

# =============================================================================
# Error Handling Class
# =============================================================================
class MPError(Exception):
    def __init__(self, message):
        """
        Format the error message in a box to make it clear this
        was a expliclty raised exception.
        """
        msg = '\n+'+'-'*78+'+'+'\n' + '| multiPointSparse Error: '
        i = 25
        for word in message.split():
            if len(word) + i +1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 2 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(79-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        print(msg)
        Exception.__init__(self)

# =============================================================================
# Utility create groups function
# =============================================================================

def createGroups(sizes, comm):
    """
    Create groups takes a list of sizes, and creates new MPI
    communicators coorsponding to those sizes. This is typically used
    for generating the communicators for an aerostructural analysis. 

    Parameters
    ----------
    sizes : list or array
        List or integer array of the sizes of each split comm
    comm : MPI intracomm
        The communicator to split. comm.size must equal sum(sizes)
        """

    nGroups = len(sizes)
    nProc_total  = sum(sizes)
    if not(comm.size == nProc_total):
        raise MPError('Cannot split comm. Comm has %d processors, but\
        requesting to split into %d.'%(comm.size, nProc_total))

    # Create a cumulative size array
    cumGroups = [0]*(nGroups+1)
    cumGroups[0] = 0   

    for igroup in xrange(nGroups):
        cumGroups[igroup+1] = cumGroups[igroup] + sizes[igroup]

    # Determine the member_key for each processor
    for igroup in xrange(nGroups):
        if comm.rank >= cumGroups[igroup] and \
               comm.rank < cumGroups[igroup+1]:
            member_key = igroup

    new_comm = comm.Split(member_key)

    flags = [False]*nGroups
    flags[member_key] = True

    return new_comm, flags

def redirectIO(f):
    """
    Redirect stdout/stderr to the given file handle.
    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.
    Written by Bret Naylor
    """
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Flush and close sys.stdout/err - also closes the file descriptors (fd)
    sys.stdout.close()
    sys.stderr.close()

    # Make original_stdout_fd point to the same file as to_fd
    os.dup2(f.fileno(), original_stdout_fd)
    os.dup2(f.fileno(), original_stderr_fd)

    # Create a new sys.stdout that points to the redirected fd
    sys.stdout = os.fdopen(original_stdout_fd, 'wb', 0) # 0 makes them unbuffered
    sys.stderr = os.fdopen(original_stderr_fd, 'wb', 0)

    # For Python 3.x
    # sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
    # sys.stderr = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

# =============================================================================
# MultiPoint Class
# =============================================================================
class multiPointSparse(object):
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
    multiPointSparse requires useGroups=True (default) when creating
    the optProb (Optimization instance). 
    """
    def __init__(self, gcomm):
        assert type(gcomm) == MPI.Intracomm
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
        that use the same obj() and sens() froutines. Members of
        processor sets typically, but not necessairly, return the same
        number of functions. In all cases, the function names must be
        unique. 
        
        Parameters
        ----------
        setName : str
            Name of process set. Process set names must be unique

        nMembers : int
            Number of members in the set.

        memberSizes : int, iteratable
            Number of processors on each set. If an iteger is suppled all\
            members use the same number of processors.\
            If a list or array is provided, a different number of processors\
            on each member can be specified. 

        Examples
        --------
        >>> MP.addProcessorSet('cruise', 3, 32)
        >>> MP.addProcessorSet('maneuver', 2, [10, 20])
        """
        # Lets let the user explictly set nMembers to 0. This is
        # equilivant to just turning off that proc set. 
        if nMembers == 0:
            self.dummyPSet.add(setName)
        else:
            nMembers = int(nMembers)
            memberSizes = numpy.atleast_1d(memberSizes)
            if len(memberSizes) == 1:
                memberSizes = numpy.ones(nMembers)*memberSizes[0]
            else:
                if len(memberSizes) != nMembers:
                    raise MPError('The suppliled memberSizes list is not \
     the correct length.')

            self.pSet[setName] = procSet(setName, nMembers, memberSizes,
                                         len(self.pSet))

    def createCommunicators(self):
        """
        Create the communicators after all the procSets have been
        added. All procSets MUST be added before this routine is
        called.

        Returns
        -------
        comm : MPI.Intracomm
            This is the communicator for the member of the procSet. Basically,
            this is the communciator that the (parallel) analyais should be
            created on 
        setComm : MPI.Intracomm
            This is the communicator that spans the entire processor set. 
        setFlags : dict
            This is a dictionary whose entry for \"setName\", as specified in
            addProcessorSet() is True on a processor belonging to that set. 
        groupFlags : list
            This is list is used to destinguish between members within
            a processor set. This list of of length nMembers and the
            ith entry is true for the ith group. 
        ptID : int
            This is the index of the group that this processor belongs to

        Examples
        --------
        >>> comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
        >>> # The following will be true for all processors for the second member
            # of the 'cruise' procSet'
        >>> setFlags['cruise'] and groupFlags[1] == True
        """

        # First we determine the total number of required procs:
        nProc = 0
        for setName in self.pSet:
            nProc += self.pSet[setName].nProc

        # Check the sizes
        if nProc < self.gcomm.size or nProc > self.gcomm.size:
            raise MPError('multiPointSparse must be called with EXACTLY\
 %d processors.'% (nProc))

        # Create a cumulative size array
        setCount = len(self.pSet)
        setSizes = numpy.zeros(setCount)
        for setName in self.pSet:
            setSizes[self.pSet[setName].setID] = self.pSet[setName].nProc
        
        cumSets = numpy.zeros(setCount+1,'intc')
        for i in range(setCount):
            cumSets[i+1] = cumSets[i] + setSizes[i]

        setFlags = {}

        # Determine the member_key for each processor
        for key in self.pSet:
            if self.gcomm.rank >= cumSets[self.pSet[key].setID] and \
                    self.gcomm.rank < cumSets[self.pSet[key].setID+1]:
                memberKey = self.pSet[key].setID
                setFlags[self.pSet[key].setName] = True
            else:
                setFlags[self.pSet[key].setName] = False

        setComm = self.gcomm.Split(memberKey)

        # Set this new_comm into each pSet and let each procSet create
        # its own split:
        for key in self.pSet:
            if setFlags[key]:

                self.pSet[key].gcomm = setComm
                self.pSet[key].createCommunicators()

                self.gcomm.barrier()

                comm = self.pSet[key].comm
                groupFlags = self.pSet[key].groupFlags
                ptID = self.pSet[key].groupID

        self.setFlags = setFlags
        # Now just append the dummy procSets:
        for key in self.dummyPSet:
            self.setFlags[key] = False
            
        self.pSetRoot = {}
        for key in self.pSet:
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
        
        for iset in self.setFlags:
            if self.setFlags[iset]:
                return iset

    def createDirectories(self, rootDir):

        """
        This function can be called only after all the procSets have
        been added. This can facilitate distingushing output files
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
        {'cruise': ['/home/user/output/cruise_0','/home/user/output/cruise_1',
                    '/home/user/output/cruise_2'],
         'maneuver':['/home/user/output/maneuver_0','/home/user/output/maneuver_1']}
         """
            
        if len(self.pSet) == 0: 
            return

        ptDirs = {}
        for key in self.pSet:
            ptDirs[key] = []
            for i in range(self.pSet[key].nMembers):
                dirName = rootDir + '/%s_%d'% (self.pSet[key].setName, i)
                ptDirs[key].append(dirName)

                if self.gcomm.rank == 0: # Only global root proc makes
                                         # directories
                    os.system('mkdir -p %s'%(dirName))
                 
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
            raise MPError("setName '%s' has not been added with "
                          "addProcessorSet."%setName)
        if not isinstance(func, types.FunctionType):
            raise MPError('func must be a Python function handle.')

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
            raise MPError("setName '%s' has not been added with "
                          "addProcessorSet."%setName)
        if not isinstance(func, types.FunctionType):
            raise MPError('func must be a Python function handle.')

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
            raise MPError("setName '%s' has not been added with "
                          "addProcessorSet."%setName)
        if not isinstance(func, types.FunctionType):
            raise MPError('func must be a Python function handle.')

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
            raise MPError("setName '%s' has not been added with "
                          "addProcessorSet."%setName)
        if not isinstance(func, types.FunctionType):
            raise MPError('func must be a Python function handle.')

        self.pSet[setName].sensFunc.append(func)
        
    def setObjCon(self, func):
        """
        Set the python function handle to compute the final objective
        and constriaints that are combinations of the functionals.

        Parameters
        ----------
        func : Python function
            Python function handle 
            """
        if not isinstance(func, types.FunctionType):
            raise MPError('func must be a Python function handle.')

        # Also do some checking on function prototype to make sure it
        # is ok:
        argSpec = inspect.getargspec(func)
        if (argSpec.varargs is not None or
            argSpec.keywords is not None or
            argSpec.defaults is not None or
            len(argSpec.args) not in [1, 2, 3]):
            raise MPError("The function signature for the function given "
                          "to 'setObjCon' is invalid. It must be: "
                          "def objCon(funcs):, def objCon(funcs, printOK): "
                          "or def objCon(funcs, printOK, passThroughFuncs):")
        
        # Now we know that there are exactly one or two arguments.
        self.nUserObjConArgs = len(argSpec.args)
        self.userObjCon = func
       
    def setOptProb(self, optProb):
        """
        Set the optimization problem that this multiPoint object will
        be used for. This is required for this class to know how to
        assemble the gradients. If the optProb is not 'finished', it
        will done so here. Therefore, this function is collective on
        the comm that optProb is built on. multiPoint sparse does
        *not* hold a reference to optProb so no additional cahnges can
        be made to optProb after this function is called. 
        
        Parameters
        ----------
        optProb : pyOptSparse optimization problem class
            The optProb object to use 
            """
       
        optProb.finalizeDesignVariables()
        optProb.finalizeConstraints()

        # Since there is no distinction between objective(s) and
        # constraints just put everything in conKeys, including the
        # objective(s)
        for iCon in optProb.constraints:
            if not optProb.constraints[iCon].linear:
                self.conKeys.add(iCon)
                self.outputWRT[iCon] = optProb.constraints[iCon].wrt
                self.outputSize[iCon] = optProb.constraints[iCon].ncon
        for iObj in optProb.objectives:
            self.conKeys.add(iObj)
            self.outputWRT[iObj] = list(optProb.variables.keys())
            self.outputSize[iObj] = 1

        for dvGroup in optProb.variables:
            ss = optProb.dvOffset[dvGroup]
            self.dvSize[dvGroup] = ss[1] - ss[0]
            
        self.conKeys = set(self.conKeys)

        # Check the dvsAsFuncs names to make sure they are *actually*
        # design variables and raise error
        for dv in self.dvsAsFuncs:
            if dv not in optProb.variables:
                raise MPError("The supplied design variable '%s' in "
                              "addDVsAsFunctions() call does not exist "
                              "in the supplied Optimization object."%dv)

    def addDVsAsFunctions(self, dvs):
        """This function allows you to specify a list of design variables to
        be explictly used as functions. Essentially, we just copy the
        values of the DVs directly into keys in 'fucncs' and
        automatically generate an identity jacobian. This allows the
        remainder of the objective/sensitivity computations to be
        proceeed as per usual. 

        Parameters
        ----------
        dvs : string or list of strings
           The DV names the user wants to use directly as functions
        """
        
        if type(dvs) == str:
            self.dvsAsFuncs.append(dvs)
        elif type(dvs) == list:
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
        
        if type(cons) == str:
            self.consAsInputs.append(cons)
        elif type(cons) == list:
            self.consAsInputs.extend(cons)
        
    def obj(self, x):

        """
        This is a built-in objective function that is designed to be
        used directly as an objective function with pyOptSparse. The
        user should not use this function directly, instead see the
        class documentation for the inteded usage. 

        Parameters
        ----------
        x : dict
            Dictionary of variables returned from pyOptSparse
        """
        for key in self.pSet:
            if self.setFlags[key]: 
                # Run "obj" funtion to generate functionals
                res = {'fail':False}
                for func in self.pSet[key].objFunc:
                    tmp = func(x)
                    if tmp is None:
                        raise MPError("No return from user supplied objective "
                                      "function for pSet %s. Functional "
                                      "derivatives must be returned in a "
                                      "dictionary."% key)

                    if 'fail' in tmp:
                        res['fail'] = bool(tmp.pop('fail') or res['fail'])
                    res.update(tmp)
                    
        if self.objCommPattern is None:
            # On the first pass we need to determine the (one-time)
            # communication pattern

            # Send all the keys
            allKeys = self.gcomm.allgather(list(res.keys()))
           
            self.objCommPattern = dict()  

            for i in range(len(allKeys)): # This is looping over processors
                for key in allKeys[i]: # This loops over keys from proc
                    if key not in self.objCommPattern:
                        if key != 'fail':
                            # Only add on the lowest proc and ignore on higher
                            # ones
                            self.objCommPattern[key] = i
              
        # Perform Communication of functionals
        allFuncs = dict()
        for key in self.objCommPattern:
            if self.objCommPattern[key] == self.gcomm.rank:
                tmp = self.gcomm.bcast(res[key], root=self.objCommPattern[key])
            else:
                tmp = self.gcomm.bcast(None, root=self.objCommPattern[key])

            allFuncs[key] = tmp
          
        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res['fail'], op=MPI.LOR)
        
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
        self.inputKeys = funckeys.difference(self.conKeys)   # input = func - con
        self.outputKeys = self.conKeys.difference(funckeys)  # output = con - func
        self.passThroughKeys = funckeys.intersection(self.conKeys)   # passThrough = func & con

        # Manage any keys that are both inputs and constraints (consAsInputs)
        # Check consAsFuncs only contains keys contained in passThoughKeys
        # inputKeys += consAsInputs
        # passThroughKeys -= consAsInputs
        if len(self.consAsInputs) > 0:
            self.consAsInputs = set(self.consAsInputs)
            self.consAsInputs.intersection_update(self.passThroughKeys)
            self.inputKeys.update(self.consAsInputs)
            self.passThroughKeys.difference_update(self.consAsInputs)  
               
        inputFuncs = self._extractKeys(allFuncs, self.inputKeys)
        passThroughFuncs = self._extractKeys(allFuncs, self.passThroughKeys)
        funcs = self._userObjConWrap(inputFuncs, True, passThroughFuncs)
        
        # Add the pass-through ones back:
        funcs.update(passThroughFuncs)

        (funcs, fail) = self.gcomm.bcast((funcs, fail), root=0)

        return funcs, fail
    
    def sens(self, x, funcs):
        """
        This is a built-in sensitity function that is designed to be
        used directly as a the sensitivty function with
        pyOptSparse. The user should not use this function directly,
        instead see the class documentation for the intended usage. 

        Parameters
        ----------
        x : dict
            Dictionary of variables returned from pyOptSparse
        """
        for key in self.pSet:
            if self.setFlags[key]: 
                # Run "sens" funtion to functionals sensitivities
                res = {'fail':False}
                for func in self.pSet[key].sensFunc:
                    tmp = func(x, funcs)
                    if tmp is None:
                        raise MPError("No return from user supplied sensitivity "
                                      "function for pSet %s. Functional "
                                      "derivatives must be returned in a "
                                      "dictionary."% key)
                    if 'fail' in tmp:
                        res['fail'] = bool(tmp.pop('fail') or res['fail'])

                    res.update(tmp)

        if self.sensCommPattern is None:
            # On the first pass we need to determine the (one-time)
            # communication pattern

            # Send all the keys
            allKeys = self.gcomm.allgather(list(res.keys()))
           
            self.sensCommPattern = dict()  

            for i in range(len(allKeys)): # This is looping over processors
                for key in allKeys[i]: # This loops over keys from proc
                    if key not in self.sensCommPattern:
                        if key != 'fail':
                            # Only add on the lowest proc and ignore on higher
                            # ones
                            self.sensCommPattern[key] = i

        # Perform Communication of functional (derivatives)
        funcSens = dict()
        for key in self.sensCommPattern:
            if self.sensCommPattern[key] == self.gcomm.rank:
                tmp = self.gcomm.bcast(res[key], root=self.sensCommPattern[key])
            else:
                tmp = self.gcomm.bcast(None, root=self.sensCommPattern[key])
 
            funcSens[key] = tmp
           
        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res['fail'], op=MPI.LOR)

        # Add in the sensitivity of the extra DVs as Funcs...This will
        # just be an identity matrix
        for dv in self.dvsAsFuncs:
            if numpy.isscalar(x[dv]):
                funcSens[dv] = {dv:numpy.eye(1)}
            else:
                funcSens[dv] = {dv:numpy.eye(len(x[dv]))}

        # Now we have to perform the CS loop over the user-supplied
        # objCon function to generate the derivatives of our final
        # constraints (and objective(s)) with respect to the
        # intermediate functionals. We will put everything in gcon
        # (including the objective)

        gcon = {}
        # Extract/Complexify just the keys we need:
        passThroughFuncs = self._extractKeys(self.funcs, self.passThroughKeys)
        cFuncs = self._extractKeys(self.funcs, self.inputKeys)
        cFuncs = self._complexifyFuncs(cFuncs, self.inputKeys)

        # Just copy the passthrough keys and keys that are both inputs and constrains:
        for pKey in self.passThroughKeys:
            gcon[pKey] = funcSens[pKey]
        for cKey in self.consAsInputs:
            gcon[cKey] = funcSens[cKey]

        # Setup zeros for the output keys:
        for oKey in self.outputKeys:
            gcon[oKey] = {}
            # Only loop over the DVsets that this constraint has:
            for dvSet in self.outputWRT[oKey]:
                gcon[oKey][dvSet] = numpy.zeros(
                    (self.outputSize[oKey], self.dvSize[dvSet]))

        for iKey in self.inputKeys: # Keys to peturb:
            if numpy.isscalar(cFuncs[iKey]):
                cFuncs[iKey] += 1e-40j
                con = self._userObjConWrap(cFuncs, False, passThroughFuncs)
                cFuncs[iKey] -= 1e-40j

                # Extract the derivative of output key variables 
                for oKey in self.outputKeys: 
                    n = self.outputSize[oKey] 
                    for dvSet in self.outputWRT[oKey]:
                        if dvSet in funcSens[iKey]:
                            deriv = (numpy.imag(con[oKey])/1e-40).reshape((n, 1))
                            gcon[oKey][dvSet] += numpy.dot(
                                deriv, numpy.atleast_2d(funcSens[iKey][dvSet]))

            else:
                for i in range(len(cFuncs[iKey])):
                    cFuncs[iKey][i] += 1e-40j
                    con = self._userObjConWrap(cFuncs, False, passThroughFuncs)
                    cFuncs[iKey][i] -= 1e-40j
                    
                    # Extract the derivative of output key variables 
                    for oKey in self.outputKeys: 
                        n = self.outputSize[oKey]
                        for dvSet in self.outputWRT[oKey]:
                            if dvSet in funcSens[iKey]:
                                deriv = (numpy.imag(con[oKey])/1e-40).reshape((n, 1))
                                gcon[oKey][dvSet] += \
                                    numpy.dot(deriv, numpy.atleast_2d(
                                        funcSens[iKey][dvSet][i, :]))

        gcon = self.gcomm.bcast(gcon, root=0)
        fail = self.gcomm.bcast(fail, root=0)

        return gcon, fail

    def _complexifyFuncs(self, funcs, keys):
        """ Convert functionals to complex type"""
        for key in keys:
            if not numpy.isscalar(funcs[key]):
                funcs[key] = numpy.array(funcs[key]).astype('D')

        return funcs

    def _extractKeys(self, funcs, keys):
        """Return a copy of the dict with just the keys given in keys"""
        newDict = {}
        for key in keys:
            newDict[key] = copy.deepcopy(funcs[key])
        return newDict

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
            
class procSet(object):
    """
    A container class to bundle information pretaining to a specific
    processor set. It is not intended to be used externally by a user.
    No error checking is performed since the multiPoint class should
    have already checked the inputs.
    """
    def __init__(self, setName, nMembers, memberSizes, setID):
        self.setName = setName
        self.nMembers = nMembers
        self.memberSizes = memberSizes
        self.nProc = numpy.sum(self.memberSizes)
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
        cumGroups = numpy.zeros(self.nMembers + 1,'intc')

        for i in range(self.nMembers):
            cumGroups[i+1] = cumGroups[i] + self.memberSizes[i]

        # Determine the member_key (m_key) for each processor
        m_key = None
        for i in range(self.nMembers):
            if (self.gcomm.rank >= cumGroups[i] and
                self.gcomm.rank < cumGroups[i+1]):
                m_key = i
                
        self.comm = self.gcomm.Split(m_key)
        self.groupFlags = numpy.zeros(self.nMembers, bool)
        self.groupFlags[m_key] = True
        self.groupID = m_key
        self.cumGroups = cumGroups
        
