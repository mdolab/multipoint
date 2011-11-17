import multiPoint
import numpy
from mdo_import_helper import MPI, mpiPrint

def obj_Aero_high(x):
    res = {}
    res['cl'] = 2.0 + MP.pSet['Aero_high'].groupID
    res['cd'] = 0.1 + MP.pSet['Aero_high'].groupID

    return res

def sens_Aero_high(x):
    pass

def obj_Aero_low(x):
    res = {}
    res['cd_low'] = 0.01 + comm.rank

    return res

def sens_Aero_low(x):
    pass

def obj_Fric(x):
    res = {}
    res['cd0'] = 10.0 + comm.rank

    return res

def sens_Fric(x):
    pass

def objective(funcs):
    # There are 2 cd's from aeroHigh, 3 cd_low's and 1
    # Fric. Average aero_high and aero_low and then add fric:

    avgDragHigh = numpy.average(funcs['cd'])
    avgDragLow  = numpy.average(funcs['cd_low'])

    total_drag = avgDragHigh + avgDragLow + funcs['cd0']

    return total_drag

def constraints(funcs):
    # Assemble all the constraint functions:
    f_con = []
    f_con.extend(funcs['cl'])

    return f_con

# Setup the multiPoint Object
MP = multiPoint.multiPoint(MPI.COMM_WORLD)
MP.addProcessorSet("Aero_high", 2, 2, {'cl':True, 'cd':True})
MP.addProcessorSet("Aero_low", 3, 1, {'cd_low':True}) 
MP.addProcessorSet("Fric", 1, 1, {'cd0':False}) #

# Create the communicators. This is must be done before the solver
# objects are created. 
setComm, comm, setFlags, groupFlags = MP.createCommunicators()

print 'myid:', MPI.COMM_WORLD.rank, setComm.rank, comm.rank, setFlags, \
    groupFlags

# Set the objective/Sens functions:
MP.setObjFunc("Aero_high", obj_Aero_high)
MP.setObjFunc("Aero_low", obj_Aero_low)
MP.setObjFunc("Fric", obj_Fric)

MP.setSensFunc("Aero_high", sens_Aero_high)
MP.setSensFunc("Aero_low", sens_Aero_low)
MP.setSensFunc("Fric", sens_Fric)

# Set the Objective Function that computes the actual objective
# from the computed functionals:

MP.setObjectiveFunction(objective)
MP.setConstraintsFunction(constraints)
#MP.setSensitiviy(sensitivity) # Not strictly necessary...the
#sensitivty of the objective wrt to the functionals can be
#computed with CS if sensitivity is not set.

x = {}
MP.fun_obj(x)
