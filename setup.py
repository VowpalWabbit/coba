# type: ignore
import setuptools

MAJOR               = 3
MINOR               = 8
MICRO               = 1
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
    packages=["coba", "coba.simulations", "coba.benchmarks", "coba.learners", "coba.pipes", "coba.config"],
    entry_points={"coba.register": "coba = coba.register"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    install_requires = [
        'requests>=2',
        'importlib-metadata',
        'typing-extensions'
    ],
    python_requires=">=3.6",
)