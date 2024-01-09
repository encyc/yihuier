from setuptools import setup
from setuptools import find_packages

VERSION = '0.1.9'

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='Yihuier',  # package name
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
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Build Tools",
        "Intended Audience :: Developers"
    ],
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'ydata_profiling==4.6.3',
    ],

    # data_files=[
    #     ('', ['conf/*.conf']),
    #     ('/usr/lib/systemd/system/', ['bin/*.service']),
    # ],
    # exclude_package_data={
    #     'bandwidth_reporter': ['*.txt']
    # },
    package_data={
        'Data': ['*.csv']
    }
)
