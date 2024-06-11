from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='3DCHESS',
    version='1.0.0',
    description='Decentralized, Distributed, Dynamic, and Context-aware Heterogeneous Sensor Systems',
    author='SEAK Lab',
    author_email='aguilaraj15@tamu.edu',
    packages=['chess3d'],
    scripts=[],
    install_requires=['matplotlib', 'neo4j', 'pyzmq', 'tqdm', 'instrupy', 'orbitpy', 'dmas'] 
)
