from setuptools import setup, find_packages

setup(
    name="bayesinsight",
    version="0.1.0",
    packages=find_packages(include=["bayesinsight", "bayesinsight.*"]),
)
