# type: ignore
import setuptools

MAJOR               = 1
MINOR               = 0
MICRO               = 3
VERSION             = f"{MAJOR}.{MINOR}.{MICRO}"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="coba",
    version=VERSION,
    author="Mark Rucker",
    author_email="rucker.mark@gmail.com",
    description="A contextual bandit benchmarking package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VowpalWabbit/coba",
    license="BSD 3-Clause License",
    packages=["coba"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires=">=3.6",
)