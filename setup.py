import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smdino",
    version="1.0",
    author="Luis Aros Illanes",
    author_email="laros@uc.cl",
    description="Steerable Medical Dino",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larosi/steerable-medical-dino",
    packages=['src/smdino'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)