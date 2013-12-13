#!/usr/bin/python
"""
multiPointSparse.py -- A python utility for aiding complex multi-point
optimizations -- designed to work specifically with pyOptSparse. 

Copyright (c) 2013 by Dr. G. K. W. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 12/06/2013$


Developers:
-----------
- Dr. G. K. W. Kenway

History
-------
v. 1.0  - First implementatino
'''

__version__ = '$Revision: $'

"""

# =============================================================================
# Standard Python modules
# =============================================================================
import sys, os, types, copy
from collections import OrderedDict

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import MPI, mpiPrint

# =============================================================================
# Error Handling Class
# =============================================================================

class MPError(Exception):
   def __init__(self, message):
      """
      Format the errro message in a box to make it clear  this
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
         print msg

# =============================================================================
# MultiPoint Class
# =============================================================================
class multiPoint(object):
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
    >>> MP = multiPointSparse.multiPoint(MPI.COMM_WORLD)
    """
    def __init__(self, gcomm):
        assert type(gcomm) == MPI.Intracomm
        self.gcomm = gcomm
        self.pSet = OrderedDict() 
        self.objective = None
        self.setFlags = None
        self.constraints = None
        self.optProb = None
        self.cumSets = [0]
        self.commPattern = None
        # User-specified functions for optimization 
        self.userObjCon = None

        return

    def addProcessorSet(self, setName, nMembers, memberSizes):
        """

        A Processor set is defined as one or more groups of processors
        that use the same obj() and sens() froutines.
        
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

        nMembers = int(nMembers)
        memberSizes = numpy.atleast_1d(memberSizes)
        if len(memberSizes) == 1:
            memberSizes = numpy.ones(nMembers)*memberSizes[0]
        else:
            if len(memberSizes) != nMembers:
                raise MPError('The suppliled memberSizes list is not the correct length')
            # end if
        # end if

        self.pSet[setName] = procSet(setName, nMembers, memberSizes, len(self.pSet))

        return

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
        for setName in self.pSet.keys():
            nProc += self.pSet[setName].nProc
        # end if

        # Check the sizes
        if nProc < self.gcomm.size or nProc > self.gcomm.size:
            mpiPrint('Error: multiPoint must be called iwth EXACTLY\
 %d processors'% (nProc), comm=self.gcomm)
            sys.exit(1)
        # end if

        # Create a cumulative size array
        setCount = len(self.pSet)
        setSizes = numpy.zeros(setCount)
        for setName in self.pSet.keys():
            setSizes[self.pSet[setName].setID] = self.pSet[setName].nProc
        # end if
        
        cumSets = numpy.zeros(setCount+1,'intc')
        for i in xrange(setCount):
            cumSets[i+1] = cumSets[i] + setSizes[i]
        # end for

        setFlags = {}

        # Determine the member_key for each processor
        for key in self.pSet.keys():
            if self.gcomm.rank >= cumSets[self.pSet[key].setID] and \
                    self.gcomm.rank < cumSets[self.pSet[key].setID+1]:
                member_key = self.pSet[key].setID
                setFlags[self.pSet[key].setName] = True
            else:
                setFlags[self.pSet[key].setName] = False
            # end if
        # end for

        setComm = self.gcomm.Split(member_key)

        # Set this new_comm into each pSet and let each procSet create
        # its own split:
        for key in self.pSet.keys():
            if setFlags[key]:

                self.pSet[key].gcomm = setComm
                self.pSet[key]._createCommunicators()

                self.gcomm.barrier()

                comm = self.pSet[key].comm
                groupFlags = self.pSet[key].groupFlags
                pt_id = self.pSet[key].groupID
            # end if
        # end for

        self.setFlags = setFlags
        
        self.pSetRoot = {}
        for key in self.pSet:
            self.pSetRoot[key] = cumSets[self.pSet[key].setID]
        # end if

        return comm, setComm, setFlags, groupFlags, pt_id

    def createDirectories(self, rootDir):
        """
        This function must be called after all the procSets have been
        added.  This can facilitate distingushing output files when
        there are a large number of points

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
            mpiPrint('Warning: No processorSets added. Cannot create \
directories',comm=self.gcomm)
            return
        # end if

        pt_dirs = {}
        for key in self.pSet.keys():
            pt_dirs[key] = []
            for i in xrange(self.pSet[key].nMembers):
                dir_name = root_dir + '/%s_%d'%(self.pSet[key].setName,i)
                pt_dirs[key].append(dir_name)

                if self.gcomm.rank == 0: # Only global root proc makes directories
                    os.system('mkdir -p %s'%(dir_name))
                # end if
            # end for
        # end for
                 
        return pt_dirs

    def setProcSetObjFunc(self, setName, func):
        """
        Set the python function handle to compute the functionals

        Parameters
        ----------
        setName : str
            Name of set we are setting the function for
        func : Python function
            Python function handle 
            """
        
        assert setName in self.pSet.keys(), "setName has not been added with\
 addProcessorSet"
        assert isinstance(func, types.FunctionType), "func must be a Python function."
        self.pSet[setName].objFunc = func
        
        return

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
        
        assert setName in self.pSet.keys(), "setName has not been added with\
 addProcessorSet"
        assert isinstance(func, types.FunctionType), "func must be a Python function."
        self.pSet[setName].sensFunc = func
        
        return

    def setObjCon(self, func):
        """
        Set the python function handle to compute the final objective
        and constriaints that are combinations of the functionals.

        Parameters
        ----------
        func : Python function
            Python function handle 
            """

        assert isinstance(func, types.FunctionType), "func must be a Python function."
        self.userObjCon = func
        
        return

    def setOptProb(self, optProb):
        """
        Set the optimization problem that this multiPoint object will
        be used for. This is required for this class to know how to
        assemble the gradients. The optProb must be "finished", that is
        all variables and constraints have been added.
        
        Parameters
        ----------
        optProb : pyOptSparse optimization problem class
            The optProb object to use 
            """
       
        self.optProb = optProb
       
        conKeys = []
        for iCon in self.optProb.constraints:
           if not self.optProb.constraints[iCon].linear:
              conKeys.append(iCon)
        self.conKeys = set(conKeys)
        self.funcs = None
        self.inputKeys = None
        self.outputKeys = None
        self.passThroughKeys = None
        
        return

    def obj(self, x):
        for key in self.pSet.keys():
           if self.setFlags[key]: 
              # Run "obj" funtion to generate functionals
              res = self.pSet[key].objFunc(x)

              assert res is not None,  "No return from user supplied\
 Objective function for pSet %s. Functionals must be returned in a\
 dictionary."%key

              if 'fail' not in res:
                 res['fail'] = False
              # end if
           # end if
        # end for
 
        if self.commPattern is None:
            # On the first pass we need to determine the (one-time)
            # communication pattern

            # Send all the keys
            allKeys = self.gcomm.allgather(res.keys())
           
            self.commPattern = dict()  

            for i in xrange(len(allKeys)): # This is looping over processors
                for key in allKeys[i]: # This loops over keys from proc
                    if key not in self.commPattern:
                       if key <> 'fail':
                          # Only add on the lowest proc and ignore on higher
                          # ones
                          self.commPattern[key] = i
                       # end if
                    # end if
                # end for
            # end for
        # end if
          
        # Perform Communication of functionals
        allFuncs = dict()
        for key in self.commPattern:
            if self.commPattern[key] == self.gcomm.rank:
                tmp = self.gcomm.bcast(res[key], root=self.commPattern[key])
            else:
                tmp = self.gcomm.bcast(None, root=self.commPattern[key])
            # end if
            allFuncs[key] = tmp
        # end for
          
        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res['fail'],op=MPI.LOR)
        
        # Save the functions since we need these for the derivatives
        self.funcs = copy.deepcopy(allFuncs)
  
        # Determine which additional keys are necessary:
        funckeys = set(allFuncs.keys())
        # Input Keys are the input variables to the obj/con functions
        # Output Keys are the output variables from the obj/con functions
        self.inputKeys = funckeys.difference(self.conKeys)
        self.outputKeys = self.conKeys.difference(funckeys)
        self.passThroughKeys = funckeys.intersection(self.conKeys)

        fObj, fCon = self.userObjCon(allFuncs)

        return fObj, fCon, fail
    
    def sens(self, x, fObj, fCon):
        for key in self.pSet.keys():
           if self.setFlags[key]: 
              # Run "obj" funtion to generate functionals
              res = self.pSet[key].sensFunc(x, fObj, fCon)

              assert res is not None,  "No return from user supplied\
 Sensitivity function for pSet %s. Functional derivatives must be returned in a\
 dictionary."%key

              if 'fail' not in res:
                 res['fail'] = False
              # end if
           # end if
        # end for

        # Perform Communication of functional (derivatives)
        funcSens = dict()
        for key in self.commPattern:
            if self.commPattern[key] == self.gcomm.rank:
               tmp = self.gcomm.bcast(res[key], root=self.commPattern[key])
            else:
               tmp = self.gcomm.bcast(None, root=self.commPattern[key])
 
            funcSens[key] = tmp
        # end for
           
        # Simply do an allReduce on the fail flag:
        fail = self.gcomm.allreduce(res['fail'], op=MPI.LOR)

        # Return on everything but the root
        if self.gcomm.rank <> 0:
           return 

        # Now we have to perform the CS loop over the user-supplied
        # objCon function to generate the derivatives of our final
        # constraints with respect to the intermediate functionals

        gobj = {}
        gcon = {}

        # Complexify just the keys we need:
        funcs = self._complexifyFuncs(self.funcs, self.inputKeys)

        # Just copy the passthrough keys:
        for pKey in self.passThroughKeys:
            gcon[pKey] = funcSens[pKey]

        # Setup zeros for the output keys:
        for oKey in self.outputKeys:
            gcon[oKey] = {}
            # Only loop over the DVsets that this constraint has:
            for dvSet in self.optProb.constraints[oKey].wrt:
                ss = self.optProb.dvOffset[dvSet]['n']                 
                ndvs = ss[1]-ss[0]
                ncon = self.optProb.constraints[oKey].ncon
                gcon[oKey][dvSet] = numpy.zeros((ncon, ndvs))
            # end for
        # end for

        # Just complexify the keys to be petrurbed 'inputKeys'
        funcs = self._complexifyFuncs(self.funcs, self.inputKeys)

        # Setup zeros for the gobj and gcon returns
        for dvSet in self.optProb.variables.keys():
            ss = self.optProb.dvOffset[dvSet]['n']                 
            ndvs = ss[1]-ss[0]
            gobj[dvSet] = numpy.zeros(ndvs)
        # end for

        for oKey in self.outputKeys:
            gcon[oKey] = {}
            # Only loop over the DVsets that this constraint has:
            for dvSet in self.optProb.constraints[oKey].wrt:
                ss = self.optProb.dvOffset[dvSet]['n']                 
                ndvs = ss[1]-ss[0]
                ncon = self.optProb.constraints[oKey].ncon
                gcon[oKey][dvSet] = numpy.zeros((ncon, ndvs))
            # end for
        # end for

        for iKey in self.inputKeys: # Keys to peturb:
            if numpy.isscalar(funcs[iKey]):
                funcs[iKey] += 1e-40j
                obj, con = self.userObjCon(funcs)
                funcs[iKey] -= 1e-40j

                # Extract the derivative of output key variables 
                for dvSet in funcSens[iKey].keys():
                    deriv = numpy.imag(obj)/1e-40
                    gobj[dvSet] += deriv * funcSens[iKey][dvSet]
                # end for

                # Extract the derivative of output key variables 
                for oKey in self.outputKeys: 
                    ncon = self.optProb.constraints[oKey].ncon
                    for dvSet in self.optProb.constraints[oKey].wrt:
                        if dvSet in funcSens[iKey].keys():
                            deriv = (numpy.imag(con[oKey])/1e-40).reshape((ncon,1))
                            gcon[oKey][dvSet] += numpy.dot(deriv, numpy.atleast_2d(funcSens[iKey][dvSet]))
                    # end for
                # end for
            else:
                for i in xrange(len(funcs[iKey])):
                    funcs[iKey][i] += 1e-40j
                    obj, con = self.userObjCon(funcs)
                    funcs[iKey][i] -= 1e-40j

                    # Extract the derivative of output key variables 
                    for dvSet in funcSens[iKey].keys():
                        deriv = numpy.imag(obj)/1e-40
                        gobj[dvSet] += deriv * funcSens[iKey][dvSet][i,:]
                    # end for

                    # Extract the derivative of output key variables 
                    for oKey in self.outputKeys: 
                        ncon = self.optProb.constraints[oKey].ncon
                        for dvSet in self.optProb.constraints[oKey].wrt:
                            if dvSet in funcSens[iKey].keys():
                                deriv = (numpy.imag(con[oKey])/1e-40).reshape((ncon,1))
                                gcon[oKey][dvSet] += \
                                    numpy.dot(deriv, numpy.atleast_2d(funcSens[iKey][dvSet][i,:]))
                            # end if
                        # end for
                    # end for
                # end for
            # end if
        # end for

        return gobj, gcon, fail

    def _complexifyFuncs(self, funcs, keys):
        """ Convert functionals to complex type"""
        
        for key in keys:
            if not numpy.isscalar(funcs[key]):
                funcs[key] = numpy.array(funcs[key]).astype('D')

        return funcs


class procSet(object):
    """
    A container class to bundle information pretaining to a specific
    processor set. It is not intended to be used externally by a user
    """
    
    def __init__(self, setName, nMembers, memberSizes, setID):
        """
        This class should not be used externally. No error checking is
        performed since the multiPoint class should have already
        checked the inputs.
        """

        self.setName = setName
        self.nMembers = nMembers
        self.memberSizes = memberSizes
        self.nProc = numpy.sum(self.memberSizes)
        self.gcomm = None
        self.objFunc = None
        self.sensFunc = None
        self.cumGroups = None
        self.groupID = None
        self.groupFlags = None
        self.comm = None
        self.setID = setID
        return

    def _createCommunicators(self):
        """
        Once the comm for the procSet is determined, we can split up
        this comm as well
        """
        
        # Create a cumulative size array
        cumGroups = numpy.zeros(self.nMembers + 1,'intc')

        for i in xrange(self.nMembers):
            cumGroups[i+1] = cumGroups[i] + self.memberSizes[i]
        # end for

        # Determine the member_key (m_key) for each processor
        m_key = None
        for i in xrange(self.nMembers):
            if self.gcomm.rank >= cumGroups[i] and \
                    self.gcomm.rank < cumGroups[i+1]:
                m_key = i
            # end for
        # end for
                
        self.comm = self.gcomm.Split(m_key)
        self.groupFlags = numpy.zeros(self.nMembers, bool)
        self.groupFlags[m_key] = True
        self.groupID = m_key
        self.cumGroups = cumGroups
        
        return
