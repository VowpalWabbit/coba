# type: ignore
import setuptools

MAJOR               = 0
MINOR               = 5
MICRO               = 0
VERSION             = f"{MAJOR}.{MINOR}.{MICRO}"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bbench",
    version=VERSION,
    author="Mark Rucker",
    author_email="rucker.mark@gmail.com",
    description="A bandit algorithm benchmarking package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrucker/banditbenchmark",
    license="BSD 3-Clause License",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: BSD-3 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)