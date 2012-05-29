#!/usr//bin/python
"""
tempMP.py --- An example for using the multiPoint analysis module
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
import multiPoint

# First create multipoint object on the communicator that will contain
# all multipoint processes. This is often MPI.COMM_WORLD.
MP = multiPoint.multiPoint(MPI.COMM_WORLD)

# Next add a "ProcessorSet". A ProcessorSet is a set of communicators
# that will all compute the same information. A typical 'ProcessorSet'
# may be each analysis point in a weighted sum drag minimization. Each
# aerodynamic problem will compute the same information, (like lift,
# drag and moment). We wil tell MP how many members we want of this
# ProcessorSet type as well as the size of each of the member. 

# Create processor set for group1. This requies N_GROUP_1*N_PROCS_1
# processors

N_GROUP_1 = 2 
N_PROCS_1 = 1
MP.addProcessorSet('group1',N_GROUP_1, N_PROCS_1)

# Next we tell MP what information or functionals we would like it to
# generate. The first argument is the processorSet we just added
# (group1), the second is the keyword name defining the quantity, the
# third is the rank of the data, and the fourth is whether or not the
# data is unique on each processor (unique=True (each one generates a
# different value) or not unique (unique=False) (each processor
# generates the same values). Currently rank=1,unique=True is NOT
# supported. 

MP.addFunctionals('group1', 'group1_drag', rank=0, unique=True)
MP.addFunctionals('group1', 'group1_lift', rank=0, unique=True)
MP.addFunctionals('group1', 'group1_thickness', rank=1, unique=False)

# We will now add a second processor set that will generate more
# functionals. Note that the name given to the addFunctionals command
# MUST NOT be the same as one already added. 

N_GROUP_2 = 1
N_PROCS_2 = 1

MP.addProcessorSet('group2',N_GROUP_2, N_PROCS_2)
MP.addFunctionals('group2','group2_drag',rank=0,unique=True)

# Continue adding ProcessorSets and the associated functionals for
# however many different types of analysis are required. 

# -------------------------------------------------------------------

# Now that we've given MP all the required information of how we want
# our complicated multipoint problem setup, we can use the handy
# command createCommunicators() to automatically create all the
# required communicators:

comm, setComm, setFlags, groupFlags, pt_id = MP.createCommunicators()

# comm:  is the communicator for a given member in a ProcessorSet. This
# is the lowest comm that comes out. The analysis to create the
# functionals should be created on this comm. 

# setComm: is the communicator over the set that this processor
# belongs to. This is typically not frequently used. 

# setFlags: Flags to determine which ProcessorSet you belong to.
# setFlags['group1'] will be True if the processor belongs to group1
# and False if the processor belongs to any other group. A typical way
# to using set flags is:

# if setFlags['group1']:
#    # Do stuff for group 1'
# if setFlags['group2']:
#    # Do stuff for group 2'

# groupFlags and pt_id: These are used to determine which index a
# communicator is within a ProcessorSet. pt_id is the index of this
# processor inside the ProcessorSet. groupFlags[pt_id] is True on the
# 'pt_id' member.

# -------------------------------------------------------------------

# We now must define functions that will compute the functionals for
# each processor set. 

def group1_obj(x):

    # We must compute a single value for g1_drag, g1_lift and a vector
    # of values for g1_thickness. 

    g1_drag = x['v1'] ** 2 * (pt_id + 1)
    g1_lift = x['v1'] * 2 * 3.14159 
    g1_thick = numpy.ones(5)
    
    comm_values = {'group1_lift': g1_lift,
                   'group1_drag': g1_drag,
                   'group1_thickness': g1_thick}
    return comm_values

def group2_obj(x):
    
    g2_drag = x['v2'] ** 3

    comm_values = {'group2_drag': g2_drag}

    return comm_values

# -------------------------------------------------------------------

# We now must define functions that will compute the SENSITIIVTY of
# functionals with respect a set of design variables for each
# processor set.

def group1_sens(x):

    # We must evalue the sensitivity of the required functionals with
    # respect to our design variables. Note that the MP doesn't care
    # how many design variables you have, they just have to be
    # consistent. Now single values like g1_lift are returned as
    # vectors and vectors like g1_thick are returned as matrices.

    g1_drag_deriv  = [2*x['v1']*(pt_id + 1),0]
    g1_lift_deriv  = [2*3.14159, 0]
    g1_thick_deriv = numpy.zeros(5,2)

    comm_values = {'group1_lift': g1_lift_deriv,
                   'group1_drag': g1_drag_deriv,
                   'group1_thickness': g1_thick_deriv}

    return comm_values

def group2_sens(x):
    
    g1_drag_deriv  = [0, 3*x['v2']**2]

    
    comm_values = {'group2_drag': g2_drag_deriv}
    
    return comm_values

# -------------------------------------------------------------------

# Next we must define how these functionals are going to be combined
# into our objective and constraint functions. objective and
# constraints are user defined functions that are called from MP with
# the argument 'funcs'. 'funcs' is a dictionary of all functions from
# all ProcessorSets that is now available on all processors. What
# we're now computing is how the objective and constraints are related
# to these functionals. Typically these functions are very simple and
# are entirely written in Python with just a few lines of code. 


def objective(funcs, printOK):

    # We have N_GROUP_1 drag values from group1 which we will average,
    # and then we will add the single value from g2_drag
  
    tmp = numpy.average(funcs['group1_drag'])

    total_drag = tmp + funcs['group2_drag']

    # Now simply return our objective

    return total_drag


def constraints(funcs, printOK):

    # Assemble all the constraint functions from the computed funcs:
    f_con = []
    f_con.extend(funcs['group1_lift'])
    f_con.extend(funcs['group1_thickness'])

    return f_con

# -------------------------------------------------------------------

# Finally we need to tell MP the functions we just defined for the
# functional evaluation and gradient as well as the objective and
# constraint functions. 

# Set the objective/Sens functions:
MP.setObjFunc("group1", group1_obj)
MP.setSensFunc("group1", group1_sens)

MP.setObjFunc("group2", group2_obj)
MP.setSensFunc("group2", group2_sens)

MP.setObjectiveFunction(objective)
MP.setConstraintsFunction(constraints)


# -------------------------------------------------------------------

# Now when we setup an Optimization problem with pyOpt we use:

# opt_prob = Optimization('Ex Opt',MP.fun_obj,use_groups=True)

# ....

# Run Optimization
# snopt(opt_prob, MP.sens)

x = {}
x['v1'] = 5
x['v2'] = 2

MP.fun_obj(x)
