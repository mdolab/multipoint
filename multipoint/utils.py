import os
import sys
import io
import copy
from mpi4py import MPI
import numpy as np


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
    """ Convert functionals to complex type"""
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


def dkeys(dict):
    """Utility function to return the keys of a dict in sorted order
    so that the iteration order is guaranteed to be the same. Blame
    python3 for being FUBAR'd."""

    return sorted(list(dict.keys()))


def skeys(set):
    """Utility function to return the items of a set in sorted order
    so that the iteration order is guaranteed to be the same. Blame
    python3 for being FUBAR'd."""
    return sorted(list(set))


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


def redirectIO(f):
    """
    Redirect stdout/stderr to the given file handle.
    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.
    Written by Bret Naylor
    """
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    sys.stdout.flush()
    sys.stderr.flush()

    # Flush and close sys.stdout/err - also closes the file descriptors (fd)
    sys.stdout.close()
    sys.stderr.close()

    # Make original_stdout_fd point to the same file as to_fd
    os.dup2(f.fileno(), original_stdout_fd)
    os.dup2(f.fileno(), original_stderr_fd)

    # Create a new sys.stdout that points to the redirected fd

    if sys.version_info >= (3, 0):
        # For Python 3.x
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, "wb"))
    else:
        sys.stdout = os.fdopen(original_stdout_fd, "wb", 0)  # 0 makes them unbuffered
        sys.stderr = os.fdopen(original_stderr_fd, "wb", 0)
