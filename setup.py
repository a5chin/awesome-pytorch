from setuptools import setup, find_packages
from codecs import open

exec(open("torchnet/version").read())
setup(
    name="torchnet",
    version=__version__,
    description="Automation library for Deep Learning.",
    url="https://github.com/a5chin/awesome-pytorch",
    author="a5chin",
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)
