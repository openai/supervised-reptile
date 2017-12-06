"""
Module configuration.
"""

from setuptools import setup

setup(
    name='supervised-reptile',
    version='0.0.1',
    description='Reptile for supervised meta-learning',
    long_description='Reptile for supervised meta-learning',
    url='https://github.com/openai/supervised-reptile',
    author='Alex Nichol',
    author_email='alex@openai.com',
    license='MIT',
    keywords='ai machine learning',
    packages=['supervised_reptile'],
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'Pillow>=4.0.0,<5.0.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
