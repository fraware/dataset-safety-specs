from setuptools import setup, find_packages

setup(
    name="ds-guard",
    version="0.1.0",
    description="Dataset safety guards generated from Lean predicates",
    author="Dataset Safety Specs",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0",
    ],
)