Changelog
=========

v0.2.0
------
*(unreleased)*

- Fixed bug in :func:`~mizani.bounds.censor` where a sequence of
  values with an irregular index would lead to an exception.

- Fixed boundary issues due internal loss of precision in ported
  function :func:`~mizani.utils.seq`.

- Added :func:`mizani.breaks.extended_breaks` which computes breaks
  using a modified version of Wilkinson's tick algorithm.

- Changed the default function :meth:`mizani.transforms.trans.breaks_`
  used by :class:`mizani.transforms.trans` to compute breaks from
  :func:`mizani.breaks.mpl_breaks` to
  :func:`mizani.breaks.extended_breaks`.

- :func:`mizani.breaks.timedelta_breaks` now uses
  :func:`mizani.breaks.extended_breaks` internally instead of
  :func:`mizani.breaks.mpl_breaks`.


v0.1.0
------
*(2016-06-30)*

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.57030.svg
   :target: http://dx.doi.org/10.5281/zenodo.57030

First public release
