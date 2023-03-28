from setuptools import setup

setup(
    name='heightgrid',
    version='0.1',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    description='Minimalistic grid map package for OpenAI Gym',
    packages=['heightgrid', 'heightgrid.envs_v2'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
