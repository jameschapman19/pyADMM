from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("./requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

setup(
    name='pyADMM',
    version='0',
    include_package_data=True,
    packages=find_packages(),
    url='https://github.com/jameschapman19/pyADMM',
    author='jameschapman',
    description=(
        'ADMM examples translated to python'
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email='james.chapman.19@ucl.ac.uk',
    python_requires='>=3.6',
    install_requires=REQUIRED_PACKAGES,
    test_suite='tests',
    tests_require=[],
)