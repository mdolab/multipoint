#!/usr/bin/python
"""
multiPoint.py -- A python utility for aiding complex multi-point optimizations

Copyright (c) 2011 by Mr. G. K. W. Kenway
All rights reserved. Not to be used for commercial purposes.
Revision: 1.0   $Date: 08/11/2011$


Developers:
-----------
- Mr. G. K. W. Kenway

History
-------
v. 1.0  - First implementatino
'''

__version__ = '$Revision: $'

"""

# =============================================================================
# Standard Python modules
# =============================================================================
import sys, os, time

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================
from mdo_import_helper import MPI, mpiPrint

# =============================================================================
# MultiPoint Class
# =============================================================================
class multiPoint(object):
    
    """
    multiPoint Class. 

    The is the main multiPoint analysis class. 
    
    """
    
    def __init__(self, gcomm):

        """
        Create the multiPoint class.
        
        Input Arguments:

            gcomm: Global MPI communicator from which all processor groups
                   are created. It is usually MPI_COMM_WORLD but may be 
                   another intraCommunicator that has already been created. 
                
        Output Arguments:
            None
            
            """
        
        self.gcomm = gcomm
        self.pSet = {}          # Dict of the procSet definitions
        self.setCount = 0
        self.objective = None
        self.setFlags = None
        self.constraints = None
        self.cumSets = [0]

        self.callCounter = 0
        self.evalAfterCount = 0

        # User-specified functions for optimization 
        self.objective = None
        self.constraints = None
        self.objective_sens = None
        self.constraints_sens = None

        return

    def addProcessorSet(self, setName, nMembers, memberSizes):

        """
        Create a processor set. 

        A Processor set is defined as 1 or more groups of processors
        that produce the same functional data. For example, for a
        multi-point aerodynamic optimization, you may which to run
        analysis at 3 different Mach-Cl combinations. Since each
        analysis produces the SAME functionals, say, CL and CD, these
        groups can be considered as the same set. 

        Input Arguments:
            setName: A string name to identify this processor set

            nMembers, integer: The number of members of this set. 

            memberSizes, integer or list: The communicator sizes required for
                each member. If it is an integer, they are taken to be the
                same size. If it is a list, it MUST be of length nMembers, 
                and each entry is the desired number of processors for that
                member. 

        Output Arguments:
        
            None

            """

        nMembers = int(nMembers)
        memberSizes = numpy.atleast_1d(memberSizes)
        if len(memberSizes) == 1:
            memberSizes = numpy.ones(nMembers)*memberSizes[0]
        else:
            if len(memberSizes) != nMembers:
                print 'Error: The suppliled memberSizes list is not the\
 correct length'
                sys.exit(1)
            # end if
        # end if

        pSet = procSet(setName, nMembers, memberSizes, self.setCount)

        self.pSet[setName] = pSet
        self.setCount += 1
        self.cumSets.append(self.cumSets[-1] + pSet.nProc)
        return

    def addFunctionals(self, setName, funcName, rank=0, unique=True):
        """
        Add a functional called 'funcName' to processor set
        'setName'. Rank indicates whether the value is a vector or
        scalar. Scalars have rank=0, vectors rank=1. These are the
        only ranks supported. Unique is used to indicate whether each
        member of a proc set generates a unique value and therefore
        must be communicated to other members. Typically unique is
        True, except for cases where they are known on all processors,
        such as geometric thickness constraints

        Input Arguments:
            setName, str: The name of set to add functional to. Must have been 
                          already added with addProcessorSet()
            funcName, str: The (unique) name of the functional to add
            
        Optional Arguments: 
            rank, integer:  Rank of the expected data. 0 for scalars, 1 for 
                            vector. These are the only suppored ranks
            unique, bool: True of data is unique to a member and therefore
                          must be communicated. False if each member returns
                          the same data
                          """
        # First check if setName is added:
        assert setName in self.pSet.keys(), "setName has not been added with\
 addProcessorSet"
        
        # Check that funcName is not ALREADY added:
        assert funcName not in self.pSet[setName].functionals, "%s has\
 already been added. Use another name" %(funcName)

        # Check that rank is 0 or 1
        rank = int(rank)
        assert rank in [0,1], "Rank must be 0 for scalar or 1 for vector"

        # Check unique:
        unique = bool(unique)
        assert unique in [True,False], "Unique must be True or False"

        # Now that we've checked the data...add to appropriate procSet
        self.pSet[setName]._addFunctional(funcName, rank, unique)

        return
        

    def createCommunicators(self):
        """
        Create the split communicators and the required flags to
        create a processor partition that has been specified by adding
        processor groups

        Input Arguments:
            None

        Output Arguments:
            comm, mpi intra communicator: The communicator for the processor

            setFlags, dictionary: This is a dictionary whose entry for
            "setName", as specified in addProcessorSet is true on a
            processor belonging to that set. 

            groupFlags, list: This is used to distinguish between
            groups within a processor set. This list of of length
            nMembers and the ith entry is true for the ith group. 

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

        #if setComm.rank == 0:
        #    print 'Rank, Comm size:',self.gcomm.rank, setComm.size

        # Set this new_comm into each pSet and let each procSet create
        # its own split:
        for key in self.pSet.keys():
            if setFlags[key]:

                self.pSet[key].gcomm = setComm

                #if self.pSet[key].gcomm.rank == 0:
                #    print 'Rank, subComm size:',self.gcomm.rank, self.pSet[key].gcomm.size
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

    def createDirectories(self, root_dir):
        """
        After all the processor sets have been added, we can create a
        separate output directory for each member in each set. This
        can facilitate distingushing output files when there are a
        large number of points

            Input Arguments: root_dir, str: Directory where folders are created

            Output Arguments: A dictionary of all the created directories.
                              Each dictionary entry has key defined by 
                              'setName' and contains a list of size nMembers,
                              each entry of which is the path to the created
                              directory
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

    def setObjFunc(self, setName, func):
        """
        Set the python function handle to compute the functionals expected
        from the set definition

        Input Arguments:
            setName: Name of set we are setting function for:
            func: Python funtion

        Output Arguments:
            None
            """
        
        self.pSet[setName].objFunc = func
        
        return

    def setSensFunc(self, setName, func):
        """
        Set the python function handle to compute the sensitivity of
        thefunctionals expected from the set definition

        Input Arguments:
            setName: Name of set we are setting function for:
            func: Python funtion

        Output Arguments:
            None
            """
        
        self.pSet[setName].sensFunc = func
        
        return

    def setObjectiveFunction(self, func):
        """
        Set the user supplied function to compute the objective from
        the functionals
        """

        self.objective = func

        return

    def setConstraintsFunction(self, func):
        """
        Set the user supplied function to compute the constraints from
        the functionals
        """

        self.constraints = func

        return

    def fun_obj(self, x):
        """ This function is used to call the user specified "obj"
        functions, communicate the results, call the "objective"
        function and finally return the result to the calling optimizer
        """

        # If evalAfterCount is set, fun_obj will only return nonzero  
        # after callCounter >= evalAfterCount
        if self.callCounter > 0 and self.callCounter < self.evalAfterCount:
            self.callCounter += 1

            f_obj = 0
            f_con = numpy.zeros(self.numCon)
            fail = 0
            return f_obj, f_con, fail
        # end if

        # Call all the obj functions and exchange values WITHIN each
        # proc set
        for key in self.pSet.keys():
            if self.setFlags[key]: 

                # Run "obj" funtion
                res = self.pSet[key].objFunc(x)
                
                # First check to see if anything is actually returned in the objFunc(x)
                if res == None:
                    print 'No values returned in objective function!'
                    sys.exit(1)

                # First check to see if all the functionals were
                # computed that should have been

                for func in self.pSet[key].functionals:
                    errStr = 'Error! Missing functional %s.'% (func)
                    assert func in res, errStr
                # end for

                # If the user has NOT supplied a fail flag, assume it
                # has not failed:
                if not 'fail' in res:
                    res['fail'] = 0
                # end if

                # The final set of functionals for this set
                setFunctionals = {}                

                # Next communicate the functions

                for func in self.pSet[key].functionals:
                    
                    # Determine the rank of the func:
                    r = self.pSet[key].functionals[func]['rank']
                  
                    # Check to see if it needs tobe communicated:
                    if self.pSet[key].functionals[func]['unique']: 
                        if r == 0:
                            val = numpy.zeros(self.pSet[key].nMembers)
                        else:
                            val = numpy.zeros((self.pSet[key].nMembers,
                                               len(res[func])))
                        # end if

                        for i in xrange(self.pSet[key].nMembers):
                            if self.pSet[key].groupID == i and \
                                    self.pSet[key].comm.rank == 0:
                                val[i] = self.pSet[key].gcomm.bcast(
                                    res[func],root=self.pSet[key].cumGroups[i])
                            else:
                                val[i] = self.pSet[key].gcomm.bcast(
                                    None,root=self.pSet[key].cumGroups[i])
                            # end if
                        # end for
                                
                        setFunctionals[func] = val

                    else: # Does not need to communicated...simply copy
                        if self.pSet[key].gcomm.rank == 0:
                            setFunctionals[func] = numpy.atleast_1d(res[func])
                        # end if
                    # end if
                # end for

                # Special case for fail. Simply logical OR over entire
                # pSet gcomm:
                setFunctionals['fail'] = self.pSet[key].gcomm.allreduce(
                    res['fail'],op=MPI.LOR)
            # end if
        # end for

        # Now each set has a consistent set of functional
        # (derivatives). We now broadcast the dictionaries so that the
        # values are known on all processors:
        functionals = {}

        MPI.COMM_WORLD.barrier()
        for key in self.pSet.keys():
            if self.setFlags[key] and self.pSet[key].gcomm.rank == 0:
                tmp = self.gcomm.bcast(setFunctionals,
                                root=self.cumSets[self.pSet[key].setID])
            else:
                tmp = self.gcomm.bcast(None,
                                root=self.cumSets[self.pSet[key].setID])
            # end if

            # Add in the functionals
            functionals.update(tmp)
        # end for

        # Special case for or. Simply logical OR over entire
        # MP gcomm
        functionals['fail'] = self.gcomm.allreduce(setFunctionals['fail'],
                                                   op=MPI.LOR)
        
        # Call the objective function:
        f_obj = self.objective(functionals, True)

        # Call the constraint function:
        f_con = self.constraints(functionals, True)
        self.numCon = len(f_con)

        # Save functionals
        self.functionals = self._complexifyFunctionals(functionals)

        self.callCounter += 1
        
        return f_obj, f_con, functionals['fail']

    def sens(self, x, f_obj, f_con):
        """ This function is used to call the user specified "sens"
        functions and communicate the results. If the user has
        specified a sensitivity of the objective, that will be
        called...otherwise, we'll just CS over the "objective" and
        "constraint" functions. 
        """

        # Call all the sens functions and exchange values WITHIN each
        # proc set
        for key in self.pSet.keys():
            if self.setFlags[key]: 

                # Run "sens" funtion
                res = self.pSet[key].sensFunc(x, f_obj, f_con)

                # First check to see if all the functionals were
                # computed that should have been

                for func in self.pSet[key].functionals:
                    assert func in res, 'Error! Missing functional %s.'% (func)
                # end for

                # If the user has NOT supplied a fail flag, assume it
                # has not failed:
                if not 'fail' in res:
                    res['fail'] = 0
                # end if

                # The final set of functionals for this set
                setDerivatives = {}                

                # Next communicate the functions
                for func in self.pSet[key].functionals:

                    r = self.pSet[key].functionals[func]['rank']

                    # Check to see if it needs tobe communicated:

                    if self.pSet[key].functionals[func]['unique']:
                        if r == 0:
                            val = numpy.zeros((self.pSet[key].nMembers, 
                                               len(res[func])))
                        else:
                            val = numpy.zeros((self.pSet[key].nMembers,
                                               res[func].shape[0],
                                               res[func].shape[1]))
                        # end if

                        for i in xrange(self.pSet[key].nMembers):
                            if self.pSet[key].groupID == i and \
                                    self.pSet[key].comm.rank == 0:
                                val[i] = self.pSet[key].gcomm.bcast(
                                    res[func],root=self.pSet[key].cumGroups[i])
                            else:
                                val[i] = self.pSet[key].gcomm.bcast(
                                    None,root=self.pSet[key].cumGroups[i])
                            # end if
                        # end for
                                
                        setDerivatives[func] = val

                    else: # Does not need to communicated...simply copy
                        if self.pSet[key].gcomm.rank == 0:
                            setDerivatives[func] = numpy.atleast_2d(res[func])
                        # end if
                    # end if
                # end for
                            
                # Special case for fail. Simply logical OR over entire
                # pSet gcomm:
                setDerivatives['fail'] = self.pSet[key].gcomm.allreduce(
                    res['fail'],op=MPI.LOR)
            # end if
        # end for

        # Now each set has a consistent set of functionals. We now
        # broadcast the dictionaries so that the values are known on
        # all processors:
        derivatives = {}
        for key in self.pSet.keys():
            if self.setFlags[key] and self.pSet[key].gcomm.rank == 0:
                tmp = self.gcomm.bcast(setDerivatives,
                                root=self.cumSets[self.pSet[key].setID])
            else:
                tmp = self.gcomm.bcast(None,
                                root=self.cumSets[self.pSet[key].setID])
            # end if

            # Add in the functionals
            derivatives.update(tmp)
        # end for

        # Special case for or. Simply logical OR over entire
        # MP gcomm
        derivatives['fail'] = self.gcomm.allreduce(setDerivatives['fail'],
                                                   op=MPI.LOR)
        
        # Now we have the derivative of each functional wrt each each
        # design variable in "derivatives". We now must compute g_obj
        # and g_con. This is a highly inefficient brute force complex
        # step approximation. If the number of design variables and
        # the number constraints are both less than say 1000, it
        # should take less than a second, which is most likely
        # acceptable.

        for key in self.functionals:
            if key != 'fail':
                nDV = derivatives[key].shape[1]
                break
            # end if
        # end for
        nCon = len(self.constraints(self.functionals,False))

        g_obj = numpy.zeros(nDV)
        g_con = numpy.zeros((nCon, nDV))

        for key in self.functionals:
            if key != 'fail':
                for i in xrange(len(self.functionals[key])):

                    refVal = self.functionals[key][i]
                    self.functionals[key][i] += 1e-40j

                    d_obj_df = numpy.imag(self.objective(
                            self.functionals, False))*1e40
                    d_con_df = numpy.imag(self.constraints(
                            self.functionals, False))*1e40

                    self.functionals[key][i] = refVal

                    g_obj += d_obj_df * derivatives[key][i, :]
                    for j in xrange(len(d_con_df)):
                        g_con[j, :] += d_con_df[j] * derivatives[key][i, :]
                    # end for


                # end for
            # end if
        # end for

        return g_obj, g_con, derivatives['fail']

    def setEvalAfterCount(self, dvNum):
        """
        Designed for SNOPT gradient check, setting evalAfterCount will bypass 
        all fun_obj calls (return 0s) until callCounter >= evalAfterCount
        """

        self.evalAfterCount = dvNum

        return

    def _complexifyFunctionals(self, functionals):
        """ Convert functionals to complex type"""
        
        for key in functionals:
            try:
                functionals[key] = functionals[key].astype('D')
            except:
                pass
            # end try
        # end for

        return functionals

class procSet(object):
    """
    A container class to bundle information pretaining to a specific
    processor set. It is not intended to be used externally by a user
        """
    def __init__(self, setName, nMembers, memberSizes, setCount):
        """
        Class creation
        """

        self.setName = setName
        self.nMembers = nMembers
        self.memberSizes = memberSizes
        self.nProc = numpy.sum(self.memberSizes)
        self.setID = setCount
        self.functionals = {}
        self.gcomm = None
        self.objFunc = None
        self.sensFunc = None
        self.cumGroups = None
        self.groupID = None
        self.groupFlags = None
        self.comm = None

        return

    def _addFunctional(self, funcName, rank, unique):
        """ 
        Add functinal. No error checking since this was done in MP class. 
        """
        
        self.functionals[funcName] = {'rank':rank,'unique':unique}

        return

    def _createCommunicators(self):
        """ Once the comm for the procSet is determined, we can split
        up this comm as well
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
                
        #if m_key is None:
        #print '[%d] Split is Screwed!'%(MPI.COMM_WORLD.rank)
        #print '[%d] cumGroups:'%(MPI.COMM_WORLD.rank),cumGroups
        #print '[%d] nMmembers:'%(MPI.COMM_WORLD.rank),self.nMembers
        #print '[%d] Rank     :'%(MPI.COMM_WORLD.rank),self.gcomm.rank
        #print '[%d] Size     :'%(MPI.COMM_WORLD.rank),self.gcomm.size


        self.comm = self.gcomm.Split(m_key)
        self.groupFlags = numpy.zeros(self.nMembers, bool)
        self.groupFlags[m_key] = True
        self.groupID = m_key
        self.cumGroups = cumGroups
        
        return



#==============================================================================
# mutliPoint Test
#==============================================================================
if __name__ == '__main__':
    import testMP
    
