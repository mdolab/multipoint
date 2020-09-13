import unittest
import numpy as np
import copy
from mpi4py import MPI
from multipoint import multiPointSparse
from pyoptsparse import Optimization

gcomm = MPI.COMM_WORLD


def set1_obj(x):
    rank = gcomm.rank
    g1_drag = x["v1"] ** 2 * (rank + 1)
    g1_lift = x["v1"] * 2 * 3.14159 * (rank + 1)
    g1_thick = np.ones((5, 1))
    funcs = {"set1_lift": g1_lift, "set1_drag": g1_drag, "set1_thickness": g1_thick, "fail": False}
    return funcs


def set1_sens(x, funcs):
    rank = gcomm.rank
    g1_drag_deriv = {"v1": 2 * x["v1"] * (rank + 1), "v2": 0}
    g1_lift_deriv = {"v1": 2 * 3.14159 * (rank + 1), "v2": 0}
    g1_thick_deriv = {"v1": np.zeros((5, 1)), "v2": np.zeros((5, 1))}
    funcsSens = {"set1_lift": g1_lift_deriv, "set1_drag": g1_drag_deriv, "set1_thickness": g1_thick_deriv}
    return funcsSens


def set2_obj(x):
    funcs = {}
    g2_drag = x["v2"] ** 3
    funcs = {"set2_drag": g2_drag}
    return funcs


def set2_sens(x, funcs):
    g2_drag_deriv = {"v1": 0, "v2": 3 * x["v2"] ** 2}
    funcsSens = {"set2_drag": g2_drag_deriv}
    return funcsSens


def objCon(funcs, printOK):
    tmp = np.average(funcs["set1_drag"])
    funcs["total_drag"] = tmp + funcs["set2_drag"]
    # if printOK:
    #     print(funcs)
    return funcs


# we create a fake optimization problem to test
SET_NAMES = ["set1", "set2"]
COMM_SIZES = {"set1": [1, 1], "set2": [1]}
SET_FUNC_HANDLES = {"set1": [set1_obj, set1_sens], "set2": [set2_obj, set2_sens]}
DVS = ["v1", "v2"]
SET_FUNCS = {"set1": ["set1_lift", "set1_drag", "set1_thickness"], "set2": ["set2_drag"]}
ALL_FUNCS = [i for s in sorted(SET_FUNCS.keys()) for i in SET_FUNCS[s]]
OBJECTIVE = "total_drag"
CONS = []
ALL_OBJCONS = [OBJECTIVE] + CONS


class TestMPSparse(unittest.TestCase):
    N_PROCS = 3

    def setUp(self):
        # construct MP
        self.MP = multiPointSparse(gcomm)
        for setName in SET_NAMES:
            comm_size = COMM_SIZES[setName]
            self.MP.addProcessorSet(setName, nMembers=len(comm_size), memberSizes=comm_size)

        self.comm, self.setComm, self.setFlags, self.groupFlags, self.ptID = self.MP.createCommunicators()

        for setName in SET_NAMES:
            self.MP.addProcSetObjFunc(setName, SET_FUNC_HANDLES[setName][0])
            self.MP.addProcSetSensFunc(setName, SET_FUNC_HANDLES[setName][1])

        # construct optProb
        optProb = Optimization("multipoint test", self.MP.obj)
        for dv in DVS:
            optProb.addVar(dv)
        optProb.addObj("total_drag")
        self.MP.setObjCon(objCon)
        self.MP.setOptProb(optProb)

    def test_createCommunicators(self):
        # check that setFlags have the right keys
        self.assertEqual(set(SET_NAMES), set(self.setFlags.keys()))

        # test that setName and groupFlags are correct
        setName = self.MP.getSetName()
        counter = {}
        for name in self.setFlags.keys():
            counter[name] = 0
            if name == setName:
                counter[name] += 1
                self.assertTrue(self.setFlags[name])
            else:
                self.assertFalse(self.setFlags[name])
            counter[name] = gcomm.allreduce(counter[name], MPI.SUM)
        self.assertEqual(counter["set1"], 2)
        self.assertEqual(counter["set2"], 1)

        # test groupFlags, ptID, and comm sizes
        # groupFlags should be all false except for one entry, whose index matches ptID
        self.assertEqual(self.setComm.size, len(self.groupFlags))
        self.assertEqual(self.comm.size, 1)
        self.assertTrue(self.groupFlags[self.ptID])
        # if we set the true entry to false, then the whole thing should be all false
        tmpGroupFlags = copy.copy(self.groupFlags)
        tmpGroupFlags[self.ptID] = False
        self.assertFalse(np.any(tmpGroupFlags))

    def test_obj_sens(self):
        x = {}
        x["v1"] = 5
        x["v2"] = 2

        funcs, fail = self.MP.obj(x)
        self.assertFalse(fail)
        funcsSens, fail = self.MP.sens(x, funcs)
        self.assertFalse(fail)

        # check that funcs contains all the funcs, objective, and constraints
        self.assertTrue(set(ALL_FUNCS).union(ALL_OBJCONS).issubset(funcs.keys()))
        # check that funcSens contains all the objective and constraints derivs
        self.assertEquals(set(ALL_OBJCONS), set(funcsSens.keys()))
        # check that the derivs are wrt all DVs
        for key, val in funcsSens.items():
            self.assertEquals(set(DVS), set(val.keys()))
