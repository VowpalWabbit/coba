# type: ignore
import setuptools

MAJOR               = 6
MINOR               = 2
MICRO               = 2
VERSION             = f"{MAJOR}.{MINOR}.{MICRO}"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="coba",
    version=VERSION,
    author="Mark Rucker",
    author_email="rucker.mark@gmail.com",
    description="A contextual bandit research package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VowpalWabbit/coba",
    license="BSD 3-Clause License",
    packages=["coba", "coba.environments", "coba.learners", "coba.experiments", "coba.pipes", "coba.contexts", "coba.backports", "coba.primitives"],
    entry_points={ "coba.register": ["coba = coba.register"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    install_requires = [
        'requests>=2',
        'importlib-metadata>=1.0;python_version<"3.8"',
        'typing-extensions>=4.1,<4.2;python_version<"3.8"'
    ],
    python_requires=">=3.7",
)
