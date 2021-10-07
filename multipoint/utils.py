import copy
from mpi4py import MPI
import numpy as np
from baseclasses.utils import Error


def mpiPrint(msg, comm=None):
    """
    Prints message only on the root proc of the comm.
    If no comm is specified, it is assumed to be MPI.COMM_WORLD
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(msg)


# =============================================================================
# Utility create groups function
# =============================================================================
def _complexifyFuncs(funcs, keys):
    """Convert functionals to complex type"""
    for key in skeys(keys):
        if not np.isscalar(funcs[key]):
            funcs[key] = np.array(funcs[key]).astype("D")

    return funcs


def _extractKeys(funcs, keys):
    """Return a copy of the dict with just the keys given in keys"""
    newDict = {}
    for key in skeys(keys):
        newDict[key] = copy.deepcopy(funcs[key])
    return newDict


def dkeys(d):
    """Utility function to return the keys of a dict in sorted order
    so that the iteration order is guaranteed to be the same. Blame
    python3 for being FUBAR'd."""

    return sorted(d.keys())


def skeys(s):
    """Utility function to return the items of a set in sorted order
    so that the iteration order is guaranteed to be the same. Blame
    python3 for being FUBAR'd."""
    return sorted(s)


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
    nProc_total = sum(sizes)
    if not (comm.size == nProc_total):
        raise Error(
            "Cannot split comm. Comm has %d processors, but requesting to split into %d." % (comm.size, nProc_total)
        )

    # Create a cumulative size array
    cumGroups = [0] * (nGroups + 1)
    cumGroups[0] = 0

    for igroup in range(nGroups):
        cumGroups[igroup + 1] = cumGroups[igroup] + sizes[igroup]

    # Determine the member_key for each processor
    for igroup in range(nGroups):
        if comm.rank >= cumGroups[igroup] and comm.rank < cumGroups[igroup + 1]:
            member_key = igroup

    new_comm = comm.Split(member_key)

    flags = [False] * nGroups
    flags[member_key] = True

    return new_comm, flags
