import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nanograd",
    version="0.1.0",
    author="Fabian Rudolf",
    author_email="fabian.rudolf",
    description="nano scalar-valued autogradient calculator for neural network graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabian.rudolf/nanograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
