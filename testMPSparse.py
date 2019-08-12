# Standard Python modules
# =============================================================================
from __future__ import print_function
import os, sys, copy, time

# =============================================================================
# External Python modules
# =============================================================================
import numpy, argparse

# =============================================================================
# Extension modules
# =============================================================================
from mpi4py import MPI
from baseclasses import *
from adflow import *
from pywarp import *
from pygeo import *
from pyspline import *
from . import multiPointSparse
from pyoptsparse import Optimization, pySNOPT

# ================================================================
#                   INPUT INFORMATION  
parser = argparse.ArgumentParser()
parser.add_argument("--output",help="output directory",
                    default=os.path.abspath(os.path.dirname(sys.argv[0])))
args = parser.parse_args()
output_directory = args.output

grid_file = 'nasa_wing_l3'
FFD_file = 'nasa_wing_ffd_12x8.fmt'
problem_name     = 'wing'
Area_ref = 198.716
Span_ref = 30.0
Chord_ref =  6.5
Xcg_ref = 0.0
nTwist = 8

# Define flow case information here:
flowCases = ['fc1','fc2','fc3','fc4']
nFlowCases = len(flowCases)
nGroup = 2
mach ={}
mach['fc1'] = 0.85
mach['fc2'] = 0.84
mach['fc3'] = 0.85
mach['fc4'] = 0.85

CL_star = {}
CL_star['fc1'] = 0.5
CL_star['fc2'] = 0.5
CL_star['fc3'] = 0.45
CL_star['fc4'] = 0.55

# Create MultiPointSparse object
MP = multiPointSparse.multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet('cruise', nGroup, 8)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

# ================================================================
#               Set Options for each solver
aeroOptions = {
    # Common Paramters
    'gridFile':grid_file+'.cgns',
    'probName':problem_name,
    'outputDir':output_directory,
    'isoSurface':{'shock':1.0,'vx':-.001},

    # Physics Paramters
    'smoother':'dadi',
    'equationType':'RANS',
    'turbulenceModel':'SA',
    'turbulenceOrder':'First Order',
    'coarseDiscretization':'Central plus scalar dissipation',
    'vis4':.0156,
    'vis2':0.25,
    'vis2Coarse':0.75, 
    'dissipationScalingExponent':0.67,
    'restrictionRelaxation':.8,

    # Common Paramters
    'CFL':1.5, 
    'CFLCoarse':1.25,
    'MGCycle':'2w',
    'MGStartLevel':1, # Start on coarest
    'nCyclesCoarse':500,

    # Convergence Paramters
    'L2Convergence':1e-4,
    'L2ConvergenceCoarse':1e-4,

    # Load Balance Paramters
    'blockSplitting':True,
    'loadImbalance':0.10,
       
    # Misc Paramters
    'printIterations':True,
    'printTiming':True,
    'monitorVariables':['resrho','cl','cd','cdv','cdp','resturb','yplus'],
    'surfaceVariables':['vx','vy','vz','rho','P','mach','cp','lift'],
    'numberSolutions':True,
    
   # Adjoint Paramters
    'adjointL2Convergence':1e-5,
    'approxPC': True,
    'viscPC':False,
    'ADPC':True,
    'restartAdjoint':True,
    'adjointSolver': 'gmres',
    'adjointMaxIter': 500,
    'adjointSubspaceSize':150, 
    'adjointMonitorStep': 10,
    'preconditionerSide': 'RIGHT',
    'matrixOrdering': 'RCM',
    'globalPreconditioner': 'Additive Schwartz',
    'localPreconditioner' : 'ILU',
    'ILUFill':2,
    'ASMOverlap':2,
    'innerpreconits':3,
    'outerpreconits':1,
    'useReverseModeAD':False,
    'frozenTurbulence':False,
    }

meshOptions = {'warpType':'algebraic',
               }

optOptions = {'Major feasibility tolerance':5.0e-5,
              'Major optimality tolerance':1.00e-4,
              'Verify level':0,
              'Major iterations limit':100,
              'Minor iterations limit':47500,
              'Major step limit':.01,
              'Penalty parameter': 0.0,
              'Nonderivative linesearch':None,
              'Function precision':1.0e-5,
              'Print file':output_directory + '/SNOPT_print.out',
              'Summary file':output_directory + '/SNOPT_summary.out',
              'Problem Type':'Minimize',
              }

# ================================================================
#               Setup the Free-Form Deformation Volume
# ================================================================

FFD = pyBlock.pyBlock('plot3d',file_name=FFD_file,
                      file_type='ascii',order='f',FFD=True)

# Setup curves for ref_axis
LE_sweep = 0.660739717218
x = [12.0*0.25, 2.75*0.25 + numpy.tan(LE_sweep)*30]
y = [0,0]
z = [0,30.0]

tmp = pySpline.curve(x=x,y=y,z=z,k=2)
X = tmp(numpy.linspace(0,1,nTwist))
c1 = pySpline.curve(X=X,k=2)

# =====================================================
#        Setup Design Variable Functions
# =====================================================

def twist(val,geo):
    # Set all the twist values
    for i in range(nTwist):
        geo.rot_z[0].coef[i] = val[i]
   
    return

# =====================================================
#        Setup Flow Solver and Mesh
# =====================================================
if setFlags['cruise']:
    aeroProblems = {}
    for fc in flowCases:
        flow = Flow(name='Cruise Flow',mach=mach[fc],alpha=2.0,
                    beta=0.0,liftIndex=2,altitude=35000/3.28)
        ref = Reference('Baseline Reference',Area_ref,Span_ref,Chord_ref,
                    xref=Xcg_ref)
        aeroProblems[fc] = AeroProblem(name='AeroStruct Test',flow_set=flow,ref_set=ref)

    mesh = MBMesh(grid_file,comm,meshOptions=meshOptions)
    mesh.addFamilyGroup("wing")

    CFDsolver = ADFLOW(comm=comm, options=aeroOptions, mesh=mesh)
    CFDsolver.initialize(aeroProblems['fc1'])
    CFDsolver.addAeroDV('aofa')
    CFDsolver.addLiftDistribution(150, 'z')
    pos = numpy.linspace(0,30,31)
    pos[0] = .0001
    CFDsolver.addSlices('z',pos)
    wing_coords = CFDsolver.getSurfaceCoordinates('wing')
    for fc in flowCases:
        CFDsolver.addFlowCase(fc)

# =====================================================
#        Setup Geometric Constraints
# =====================================================

# (Empty) DVConstraint Object
DVCon = DVConstraints.DVConstraints()

# Load igs file for doing projections
wing = CFDsolver.getTriangulatedMeshSurface()

if setFlags['cruise']:
    # Thickness Constraints: 11x3
    root_chord = 12.0
    break_chord = 6.5
    tip_chord = 2.75
    break_pt = numpy.tan(LE_sweep)*10.5
    tip_pt = numpy.tan(LE_sweep)*Span_ref

    # Create continuous splines for x/c=0.01, x/c=0.99 and x/c=0.15
    t = [0,0,0,1,1,1]
    le = pySpline.curve(k=3,t=t,coef=[[           0.01*root_chord, 0, 0.001],
                                      [break_pt + 0.01*break_chord, 0, 10.5],
                                      [tip_pt   + 0.01*tip_chord  , 0, 30.0]])

    te = pySpline.curve(k=3,t=t,coef=[[           0.99*root_chord, 0, 0.001],
                                      [break_pt + 0.99*break_chord, 0, 10.5],
                                      [tip_pt   + 0.99*tip_chord  , 0, 30.0]])

    spar = pySpline.curve(k=3,t=t,coef=[[           0.15*root_chord, 0, 0.001],
                                        [break_pt + 0.15*break_chord, 0, 10.5],
                                        [tip_pt   + 0.15*tip_chord  , 0, 30.0]])
    s = numpy.linspace(0,1,25)

    # Add volume constraint:
    DVCon.addVolumeConstraint(wing, le(s), te(s), nSpan=30, nChord=40,
                              lower=1.0,upper=3, scaled=True)

    # Add two lines: One at LE spar and one at the trailing edge. Don't decrease from here
    DVCon.addThicknessConstraints2D(wing,spar(s), te(s), 17, 25, 
                                    lower=1.0, scaled=True)
# end if

# Obtain the set of coordinates we need for the DVGeometry Obeject
con_coords = DVCon.getCoordinates()

# =====================================================
#        Setup Design Variable Mapping Object
# =====================================================

# Create an instance of DVGeo
DVGeo = DVGeometry.DVGeometry([c1],FFD=FFD,rot_type=5, axis='x')
DVGeo.addPointSet(wing_coords,'wing')
DVGeo.addPointSet(con_coords,'con')
DVGeo.addGeoDVGlobal('twist',numpy.zeros(nTwist),-7.50,7.5,twist)
DVGeo.addGeoDVLocal('shape',-1.0,1.0,'y')

up_ind = []
low_ind = []

# Note we CAN add "constraints" to control points that may not be added
# as above. 

for ivol in range(FFD.nVol):
    sizes = FFD.topo.l_index[ivol].shape
    for k in range(sizes[2]): # Go out the 'z' or 'k' direction
        up_ind.append(FFD.topo.l_index[ivol][0,-1,k])  # Le control points
        low_ind.append(FFD.topo.l_index[ivol][0,0,k])

        up_ind.append(FFD.topo.l_index[ivol][-1,-1,k])  # Te control points
        low_ind.append(FFD.topo.l_index[ivol][-1,0,k])
# end for

DVCon.addLeTeCon(DVGeo,up_ind,low_ind)

# =====================================================
#        Obective Function
# =====================================================

def cruiseObj(x):
    
    if MPI.COMM_WORLD.rank == 0:
        print('Fun Obj:')
        print(x)
        
    # Set geometric design variables from optimizer
    DVGeo.setValues(x, scaled=True)

    # Set CFD surface coordinates from updated DVGeometry opject
    CFDsolver.setSurfaceCoordinates('wing', DVGeo.update('wing'))
   
    funcs = {}
    funcs['fail'] = False
    for i in range(nFlowCases):
        if i%nGroup == ptID:
            fc = flowCases[i]
            aeroProblems[fc]._flows.alpha = x['alpha_'+fc]
            CFDsolver(aeroProblems[fc], 2000, flowCase=fc)
            sol = CFDsolver.getSolution()
            funcs['cl_'+fc] = sol['cl']
            funcs['cd_'+fc] = sol['cd']
            if CFDsolver.solveFailed and funcs['fail'] is False:
                funcs['fail'] = True
            # end if
        # end if
    # end for

    # Set Geometric constraint coordinates from updated DVGeometry
    # opject and evaluate
    DVCon.setCoordinates(DVGeo.update('con'))
    DVCon.evalConstraints(funcs, DVGeo)
    
    return funcs

def cruiseSens(x, fobj, fcon):
    fail = 0
    funcSens = {}

    for i in range(nFlowCases):
        if i%nGroup == ptID:
            fc = flowCases[i]
            # --------- cl Adjoint -----------
            CFDsolver.solveAdjoint('cl', flowCase=fc)
            dIdpt = CFDsolver.totalSurfaceDerivative('cl')
            funcSens['cl_'+fc] = {'geo':DVGeo.totalSensitivity(dIdpt,comm,name='wing'),
                                  'alpha_'+fc:CFDsolver.totalAeroDerivative('cl')}
    
            # --------- cd Adjoint -----------
            CFDsolver.solveAdjoint('cd')
            dIdpt = CFDsolver.totalSurfaceDerivative('cd')
            funcSens['cd_'+fc] = {'geo':DVGeo.totalSensitivity(dIdpt,comm,name='wing'),
                                  'alpha_'+fc:CFDsolver.totalAeroDerivative('cd')}
        # end if
    # end for

    # --------- DVConstraints Sensitivities ---------
    DVCon.evalJacobianConstraints(funcSens, DVGeo)

    return funcSens

def objCon(funcs):
    # Assemble the objective and any additional constraints:

    funcs['cd'] = 0.0
    for i in range(nFlowCases):
        fc = flowCases[i]
        funcs['cd'] += funcs['cd_'+fc]/nFlowCases

        # Compute the lift constraint by subtracting the desired
        # CL_star
        funcs['cl_con_'+fc] = funcs['cl_'+fc] - CL_star[fc]
    # end for

    return funcs

# =====================================================
#   Set-up Optimization Problem
# =====================================================

opt_prob = Optimization('opt', MP.obj, use_groups=True)

# Add Aero Variables
for fc in flowCases:
    opt_prob.addVar('alpha_'+fc, value=1.6, lower=0.,upper=10., scale=.1)

# Add Geo variables
opt_prob = DVGeo.addVariablesPyOpt(opt_prob)

# Constraints:
for fc in flowCases:
    opt_prob.addCon('cl_con_'+fc, type='i', lower=0.0, upper=0.0, 
                         scale=10, wrt=['geo','alpha_'+fc])

# Geometric Constraints
DVCon.addConstraintsPyOpt(opt_prob)

# Add Objective 
opt_prob.addObj('cd')

# Check opt problem:
if MPI.COMM_WORLD.rank == 0:
    print(opt_prob)
    opt_prob.printSparsity()

# The MP object needs the 'obj' and 'sens' function for each proc set,
# the optimization problem and what the objcon function is:
MP.setProcSetObjFunc('cruise', cruiseObj)
MP.setProcSetSensFunc('cruise',cruiseSens)
MP.setOptProb(opt_prob)
MP.setObjCon(objCon)

# Make Instance of Optimizer
snopt = pySNOPT.SNOPT(options=optOptions)

# Run Optimization
hist_file = output_directory + '/opt_hist'
snopt(opt_prob, MP.sens, store_hst=hist_file)
