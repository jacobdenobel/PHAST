import os
import subprocess
import platform
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.0.1"

ext = Pybind11Extension(
    "phast.phastcpp", glob("src/*cpp"), include_dirs=["src"], cxx_std=17
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
    packages=["phast"],
    package_data={
        "phast": [
            "phast/idet.npy",
        ],
    },
    zip_safe=False,
    version=__version__,
    install_requires=["matplotlib>=3.3.4", "numpy>=1.19.2", "scipy>=1.5.2"],
    include_package_data=True,
)
