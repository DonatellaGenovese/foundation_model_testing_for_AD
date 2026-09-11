#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="foundation_model_testing_for_AD",
    version="0.1.0",
    description="Explaining anomalies in collider data via learned latent representations",
    author="Donatella Genovese",
    license="MIT",
    url="https://github.com/DonatellaGenovese/foundation_model_testing_for_AD",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
