# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
    setup_requires = f.readlines()


setup(name='adversarial-nli',
      version='0.0.1',
      description='Neural Natural Language Inference',
      author='Pasquale Minervini',
      author_email='p.minervini@cs.ucl.ac.uk',
      url='https://github.com/uclmr/adversarial-nli',
      test_suite='tests',
      license='MIT',
      install_requires=setup_requires,
      extras_require={
            'tf': ['tensorflow>=1.4.0'],
            'tf_gpu': ['tensorflow-gpu>=1.4.0'],
      },
      setup_requires=setup_requires,
      tests_require=setup_requires,
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules'
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages(),
      keywords='tensorflow machine learning natural language processing')
