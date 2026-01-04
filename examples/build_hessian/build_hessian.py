#!/usr/bin/env python3
import sys
import numpy as np
import pyspawn
import pyspawn.process_geometry as pg

# terachemserver port
if len(sys.argv) < 2:
    print("Please provide a port number as a command line argument")
    print("Usage: {} <port>".format(sys.argv[0]))
    sys.exit(1)

try:
    port = int(sys.argv[1])
except ValueError:
    print("Invalid port: {!r}. Port must be an integer.".format(sys.argv[1]))
    sys.exit(1)

# Processing geometry.xyz file
natoms, atoms, pos, comment = pg.process_geometry("geometry.xyz")

# if geometry file is in Angstrom converting to Bohr
pos *= 1.889725989

print("Number of atoms =", natoms)
print("Atom labels:", atoms)
print("Positions in Bohr:\n", pos)

# choose TeraChem potential
pyspawn.import_methods.into_hessian(pyspawn.potential.terachem_dft)

# number of dimensions (3 * number of atoms)
ndims = natoms * 3

# number of electronic states
numstates = 1

# create a Hessian object
hess = pyspawn.hessian(ndims, numstates)

# select the ground state
istate = 0

# the step size for numerical Hessian calculation
dr = 0.001

# TeraChem options for B3LYP/6-31G** calculation
tc_options = {
    "method": "b3lyp",
    "basis": "6-31gss",
    "atoms": atoms,
    "charge": 0,
    "spinmult": 1,
    "closed_shell": True,
    "restricted": True,
    "precision": "double",
    "threall": 1.0e-20,
    "thregr": 1.0e-20,
    "convthre": 1.0e-08,
}

# build a dictionary containing all options
hess_options = {
    "tc_port": port,   # terachem port
    "istate": istate,
    "positions": pos,
    "tc_options": tc_options,
}

# set number of dimensions
hess.set_numdims(ndims)

# pass options to hess object
hess.set_parameters(hess_options)

# compute Hessian semianalytically (using analytic first derivatives)
hess.build_hessian_hdf5_semianalytical(dr)

