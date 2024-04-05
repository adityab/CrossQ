import os

from setuptools import find_packages, setup

with open(os.path.join("sbx", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="sbx-rl",
    packages=[package for package in find_packages() if package.startswith("sbx")],
    package_data={"sbx": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3==2.1.0",
        "flax==0.7.4",
        "gymnasium==0.29.1",
        "imageio==2.31.3",
        "mujoco==2.3.7",
        "optax==0.1.7",
        "tqdm==4.66.1",
        "rich==13.5.2",
        "rlax==0.1.6",
        "tensorboard==2.14.0",
        "tensorflow-probability==0.21.0",
        "wandb==0.15.10",
        "scipy==1.11.4",
        "shimmy==1.3.0"
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code
            "ruff",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
    },
    description="Jax version of CrossQ; Bhatt and Palenicek et al. 2023.",
    author="Aditya Bhatt, Daniel Palenicek",
    url="https://github.com/adityab/sbx-crossq",
    author_email="aditya.bhatt@dfki.de, daniel.palenicek@tu-darmstadt.de",
    keywords="crossq reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    version=__version__,
    python_requires="==3.11.5",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
)
