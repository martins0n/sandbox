from setuptools import setup, find_packages


install_requires = [
   'pandas',
   'numpy',
   'rpy2',
   'scipy',
]

package_data = {
    'pym4metalearning': ['*.R']
}

setup(
    name='pym4metalearning',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'notebooks')),
    install_requires=install_requires,
    package_data=package_data,
    include_package_data=True
)