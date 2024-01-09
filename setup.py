from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.8'

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='Yihui',  # package name
    version=VERSION,  # package version
    author='encyc',
    author_email='atomyuangao@gmail.com',
    description='Package for Logistic Regression Modeling, focus on Credit Risk Management',  # package description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/encyc/yihui',
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'ydata_profiling==4.6.3',
    ]

)

