import os
import platform
from glob import glob
from setuptools import setup, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.1.0"

ext = Pybind11Extension(
    "phast.phastcpp", glob("src/*cpp"), include_dirs=["src"], cxx_std=17
)

if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3", "-pthread"])
else:
    ext._add_cflags(["/O2"])


with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    description = f.read()

setup(
    name="phastc",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    description="Phenomological Adaptive STochastic auditory nerve fiber model",
    long_description=description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    zip_safe=False,
    version=__version__,
    install_requires=[
        "matplotlib", 
        "numpy", 
        "librosa", 
    ],
    include_package_data=True,
)
