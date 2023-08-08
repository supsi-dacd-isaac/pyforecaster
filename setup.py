import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyforecaster",
    version="0.1",
    author="Lorenzo Nespoli",
    author_email="lorenzo.nespoli@hivepower.tech",
    description="Base package for formatting and forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hivepower/pyforecaster",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3",
        "Operating System :: OS Independent",
    ],
    extras = {'graphviz':'pygraphviz>=1.10'},
    install_requires=['numpy>=1.20.2',
                      'optuna>=2.10.0',
                      'ptitprince>=0.2.5',
                      'networkx>=2.6.3',
                      'pandas>=1.2.3',
                      'seaborn>=0.11.1',
                      'matplotlib>=3.4.1',
                      'scipy>=1.7.1',
                      'scikit_learn>=1.0.1',
                      'lightgbm>=3.3.2',
                      'ray>=2.0.0',
                      'sharedmem>=0.3.8',
                      'numba>=0.56.3',
                      'holidays>=0.16',
                      'python-long-weekends>=0.1.1',
                      'jax>=0.4.1',
                      'jaxlib>=0.4.1'
                      'quantile-forest>=1.2.0'
                      ],
    python_requires='>=3.8',
)