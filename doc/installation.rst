Installation
============

mizani can be can be installed in a couple of ways depending on purpose.

Official release installation
-----------------------------
For a normal user, it is recommended to install the official release.

.. code-block:: console

    $ pip install mizani

Development installation
------------------------
To do any development you have to clone the
`mizani source repository`_ and install
the package in development mode. These commands do all of that:

.. code-block:: console

    $ git clone https://github.com/has2k1/mizani.git
    $ cd mizani
    $ pip install -e .

If you only want to use the latest development sources and do not
care about having a cloned repository, e.g. if a bug you care about
has been fixed but an official release has not come out yet, then
use this command:

.. code-block:: console

    $ pip install git+https://github.com/has2k1/mizani.git

.. _mizani source repository: https://github.com/has2k1/mizani
