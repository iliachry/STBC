from setuptools import setup, find_packages

setup(
    name="stbc",
    version="1.0.0",
    description="Space-Time Block Code (STBC) Simulation Framework",
    author="Ilias Chrysovergis",
    author_email="iliachry@iliachry.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'stbc=stbc.__main__:main',
        ],
    },
)
