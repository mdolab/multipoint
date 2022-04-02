from setuptools import setup
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("multipoint/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("doc/requirements.txt") as f:
    docs_require = f.read().splitlines()

setup(
    name="multipoint",
    version=__version__,
    description="Provides utilities to facilitates distributed multipoint optimization with the MACH framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="multi-point optimization",
    author="",
    author_email="",
    url="https://github.com/mdolab/multipoint",
    license="Apache License Version 2.0",
    packages=[
        "multipoint",
    ],
    install_requires=[
        "numpy>=1.16",
        "mpi4py>=3.0",
        "mdolab-baseclasses>=1.2.4",
    ],
    extras_require={
        "docs": docs_require,
        "testing": ["pyoptsparse>=2.5.1"],
    },
    classifiers=["Operating System :: OS Independent", "Programming Language :: Python"],
)
