pySpawn
=======

Version 1.0.0

Created by  
Benjamin G. Levine  
Stony Brook University

pySpawn is a full multiple spawning (FMS) software package written in Python.  
It is designed to be minimalistic, extensible, and suitable for ab initio nonadiabatic molecular dynamics simulations.  
The code is distributed under the MIT License.

---

Citation
========

If you use pySpawn, please cite:

https://doi.org/10.1021/acs.jctc.0c00575

If you use the OpenMolcas interface, please also cite:

https://doi.org/10.1021/acs.jctc.4c00855

---

License
=======

See the `LICENSE` file.

---

Features
========

pySpawn currently provides the following capabilities:

- Full multiple spawning in the adiabatic representation
- Derivative couplings computed via NPI
- Interface to a development version of TeraChem (via tcpb)
- Interface to OpenMolcas
- SSAIMS. Stochastic Selection AIMS (optional per run)
- Analysis tools for post-processing simulation data

The code is under active development.  
Example jobs and partial documentation are included.

---

Installation
============

pySpawn requires **Python 3** (3.8 or newer; this fork is developed and tested
under Python 3.13).  
Installation should be performed using pip inside a virtual environment to avoid dependency conflicts.

The Python dependencies are listed in `requirements.txt`:

    pip install -r requirements.txt

Option 1. Install using a virtual environment (recommended)
-----------------------------------------------------------

Step 1. Create and activate a virtual environment

    python3 -m venv pyspawn-env
    source pyspawn-env/bin/activate

Step 2. Upgrade pip

    pip install --upgrade pip

Step 3. Install pySpawn

From the root directory of the source tree:

    pip install .

Step 4. Verify the installation

    python -c "import pyspawn; print(pyspawn.__file__)"

If no error is raised, the installation is successful.

Option 2. Install into a custom directory (advanced users)
----------------------------------------------------------

This is useful on clusters or systems without write access to site-packages.

    python3 -m pip install . --target /path/to/pyspawn
    export PYTHONPATH=/path/to/pyspawn:$PYTHONPATH

To verify:

    PYTHONPATH=/path/to/pyspawn python3 -c "import pyspawn; print(pyspawn.__file__)"

---

Dependencies
============

pySpawn depends on the following Python packages (see `requirements.txt`):

- NumPy (>=1.21; tested with 2.5)
- h5py (>=3,<4)
- matplotlib (>=3; analysis / plotting)

This fork is tested with NumPy 2.x and h5py 3.x under Python 3.13; the older
NumPy upper bound noted in earlier releases is not enforced. The TeraChem and
OpenMolcas interfaces require additional software (see below).

---

Interfaces
==========

pySpawn currently provides interfaces to:

- TeraChem (development version)
- OpenMolcas

These interfaces require separate installation and configuration of the corresponding electronic structure software.


TeraChem Note
-------------

The TeraChem interface requires the **TCPB (TeraChem Protocol Buffer)** Python package, which provides client–server communication with the TeraChem engine. TCPB must be installed separately and made available in the Python environment used to run pySpawn.

TCPB can be installed using `pip`, for example:

    python3 -m pip install tcpb --target=/path/to/tcpbpy3

After installation, ensure that the target directory is included in your `PYTHONPATH` so that pySpawn can locate the TCPB package at runtime. When using the TeraChem interface, TeraChem must be launched in persistent server mode and configured to listen on the appropriate port.

---

Contact
=======

pySpawn is developed and maintained primarily by:

Benjamin G. Levine  
ben.levine@stonybrook.edu
