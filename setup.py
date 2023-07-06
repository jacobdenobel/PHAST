import os
import subprocess
import platform
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
    "phast.phastcpp", 
    glob("src/*cpp"), 
    include_dirs=["src"],
    cxx_std=17
)
if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3"])
    try:
        if subprocess.check_output("ldconfig -p | grep tbb", shell=True):
            ext._add_ldflags(["-ltbb"])
            ext._add_cflags(["-DHASTBB"])
    except subprocess.CalledProcessError:
        pass
else:
    ext._add_cflags(["/O2", "/DHASTBB"])
    
setup(
    name="phast",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    version = "0.0.1",
    dependencies = [
        "matplotlib>=3.4.2",
        "numpy>=1.19.2",
        "scipy>=1.5.2"
    ]
)