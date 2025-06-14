from setuptools import setup, find_packages

def load_requirements(filename='requirements.txt'):
    """Load requirements from a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            # Remove comments and whitespace
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name='myutils',
    version='0.1.0',
    description='Various utility functions',
    author='Peter Williams',
    author_email='pwilliams272@gmail.com',
    url='https://github.com/pwilliams272/myutils',
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)