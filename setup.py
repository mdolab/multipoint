from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("multipoint/__init__.py").read(),
)[0]

setup(
    name="multipoint",
    version=__version__,
    description="Provides utilities to facilitates distributed multipoint optimization with the MACH framework.",
    long_description="""Please see the [documentation](https://mdolab-multipoint.readthedocs-hosted.com) for API documentation.

      To locally build the documentation, enter the `doc` folder and enter `make html` in terminal.
      You can then view the built documentation in the `_build` folder.
      """,
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
        "testing": ["pyoptsparse>=2.5.1"],
    },
    classifiers=["Operating System :: OS Independent", "Programming Language :: Python"],
)
