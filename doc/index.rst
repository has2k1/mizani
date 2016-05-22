.. _index:

Mizani documentation
====================

Mizani is Python package that provides the pieces necessary to create scales for a graphics system. It is based on Hadley Wickham's `Scales`_ package.


Installation
------------

Mizani can be installed in a handful of ways.

1. Official release *(Recommended)*

   .. code-block:: console

       $ pip install mizani

   If you don't have `pip`_ installed, this `Python installation guide`_
   can guide you through the process.


2. Development sources

   .. code-block:: console

       $ pip install git+https://github.com/has2k1/mizani.git

   Or

   .. code-block:: console

       $ git clone https://github.com/has2k1/mizani.git
       $ cd mizani
       $ python setup.py install

   Or

   .. code-block:: console

       $ curl -OL https://github.com/has2k1/mizani/archive/master.zip
       $ unzip master
       $ cd mizani-master
       $ python setup.py install


Contents
--------

.. toctree::
   :maxdepth: 2

   bounds
   breaks
   formatters
   palettes
   transforms
   scale
   changelog


.. _Scales: https://github.com/hadley/scales
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
