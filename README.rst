.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/alinc.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/alinc
    .. image:: https://readthedocs.org/projects/alinc/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://alinc.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/alinc/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/alinc
    .. image:: https://img.shields.io/pypi/v/alinc.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/alinc/
    .. image:: https://img.shields.io/conda/vn/conda-forge/alinc.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/alinc
    .. image:: https://pepy.tech/badge/alinc/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/alinc
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/alinc

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
ALINC
=====


    Active Learning for Inductive Node Classification




Installation
============

The project dependency files target Python 3.13 with the newest PyTorch/PyG
binary stack that still provides a matching ``torch-scatter`` wheel:
PyTorch 2.11.0 + CUDA 12.8, TorchVision 0.26.0, PyG 2.7.0, and
``torch-scatter`` 2.1.2 for PyTorch 2.11/CUDA 12.8.

Create the environment with:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate alinc-github

or install into an existing Python 3.13 environment with:

.. code-block:: bash

   pip install -r requirements.txt


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
