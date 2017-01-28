Changelog
=========

v0.2.1
------
*(unreleased)*

- Added :func:`~mizani.palettes.xkcd_palette`, a palette that
  selects from 954 named colors.

- Added :func:`~mizani.palettes.crayon_palette`, a palette that
  selects from 163 named colors.

- Added :func:`cubehelix_pal`, a function that creates a continuous
  palette from the cubehelix system.



v0.2.0
------
*(2017-01-27)*

.. image:: https://zenodo.org/badge/62319878.svg
   :target: https://zenodo.org/badge/latestdoi/62319878

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

- Added manual palette function :func:`mizani.palettes.manual_pal`.

- Requires `pandas` version 0.19.0 or higher.

v0.1.0
------
*(2016-06-30)*

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.57030.svg
   :target: http://dx.doi.org/10.5281/zenodo.57030

First public release
