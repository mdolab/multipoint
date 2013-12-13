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
import sys, os, types
from collections import OrderedDict

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================
from mpi4py import MPI

# =============================================================================
# Error Handling Class
# =============================================================================

class MPError(Exception):
   def __init__(self, message):
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

        self.pSet[setName] = procSet(setName, nMembers, memberSizes)

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
        if nProc <> self.gcomm.size:
            raise MPError('multiPoint must be called iwth EXACTLY\
 %d processors'% (nProc))
        # end if

        # Create a cumulative size array
        setSizes = numpy.zeros(self.setCount)
        for setName in self.pSet.keys():
            setSizes[self.pSet[setName].setID] = self.pSet[setName].nProc
        # end if
        
        cumSets = numpy.zeros(self.setCount+1,'intc')
        for i in xrange(self.setCount):
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
            self.pSetRoot[key] = self.cumSets[self.pSet[key].setID]
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
            
        if len(self.pSet) == 0 and self.gcomm.rank == 0:
            print 'Warning: No processorSets added. Cannot create directories'
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
        assert isinstance(func, types.FuntionType), "func must be a Python function."
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
        assert isinstance(func, types.FuntionType), "func must be a Python function."
        self.pSet[setName].objFunc = func
        
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

        assert isinstance(func, types.FuntionType), "func must be a Python function."
        self.objcon = func
        
        return

    def fun_obj(self, x):
        
        return f_obj, f_con, functionals['fail']

    def sens(self, x, f_obj, f_con):

        return g_obj_summed, g_con_summed, derivatives['fail']


class procSet(object):
    """
    A container class to bundle information pretaining to a specific
    processor set. It is not intended to be used externally by a user
    """
    
    def __init__(self, setName, nMembers, memberSizes):
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
