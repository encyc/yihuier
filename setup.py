from setuptools import setup
from setuptools import find_packages


VERSION = '0.1.2'

with open("README.md", "r") as fh:
    long_description = fh.read

setup(
    name='Yihui',  # package name
    version='0.1.1',  # package version
    author='encyc',
    author_email='atomyuangao@gmail.com',
    description='Package for Logistic Regression Modeling, focus on Credit Risk Management',  # package description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/encyc/yihui',
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3.6'

)

