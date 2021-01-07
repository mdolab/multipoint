"""
This example demonstrates how to add processor sets and perform operations
on a specific group of processors within a set.

To run locally do: ``mpirun -np 10 --oversubscribe python ex1.py``.
"""
# rst init (begin)
# ==============================================================================
# Import modules
# ==============================================================================
from mpi4py import MPI
from multipoint import multiPointSparse

# rst init (end)
# ==============================================================================
# Processor allocation
# ==============================================================================
# Instantiate the multipoint object
MP = multiPointSparse(MPI.COMM_WORLD)

# Add all processor sets and create the communicators
MP.addProcessorSet("codeA", 3, [3, 2, 1])
MP.addProcessorSet("codeB", 1, 4)
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()

# Extract setName on a given processor for convenience
setName = MP.getSetName()

# Create all directories for all groups in all sets
ptDirs = MP.createDirectories("./output")

# For information, print out all values on all processors
print(
    f"setName={setName}, comm.rank={comm.rank}, comm.size={comm.size}, setComm.rank={setComm.rank}, setComm.size={setComm.size}, setFlags={setFlags}, groupFlags={groupFlags}, ptID={ptID}"
)
# rst alloc (end)
# ==============================================================================
# Problem setup
# ==============================================================================
# To perform operations on all processors in a set we can use the setFlags
if setFlags["codeA"]:  # Alternatively, setName == "codeA" could be used here
    # ...
    # To access a particular group within the set can be done using the ptID
    # Here we access only the processors in the first group
    if 0 == ptID:
        print(f"setName={setName} comm.rank={comm.rank} ptID={ptID}")

    # To access all groups (but still a specific one) we simply loop over the size of the set
    for i in range(setComm.size):
        if i == ptID:
            print(f"setName={setName} comm.rank={comm.rank} ptID={ptID} i={i}")

# Similarly, for the other processor set
if setFlags["codeB"]:
    for i in range(setComm.size):
        if i == ptID:
            print(f"setName={setName} comm.rank={comm.rank} ptID={ptID} i={i}")
# rst problem (end)
