import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bbench",
    version="0.0.1",
    author="Mark Rucker",
    author_email="rucker.mark@gmail.com",
    description="A package for benchmarking bandit algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrucker/banditbenchmark",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)