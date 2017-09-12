Changelog
=========

v0.3.4
------
*(not yet released)*

- Fixed issue where some formatters methods failed if passed empty
  ``breaks`` argument.


v0.3.3
------
*(2017-08-30)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.854777.svg
   :target: https://doi.org/10.5281/zenodo.854777

- Fixed *SettingWithCopyWarnings* in :func:`~mizani.bounds.squish_infinite`.
- Added :func:`~mizani.formatters.log_format`.

API Changes
***********

- Added :class:`~mizani.transforms.log_trans` now uses
  :func:`~mizani.formatters.log_format` as the formatting method.


v0.3.2
------
*(2017-07-14)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.827406.svg
   :target: https://doi.org/10.5281/zenodo.827406

- Added :func:`~mizani.bounds.expand_range_distinct`

v0.3.1
------
*(2017-06-22)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.815721.svg
   :target: https://doi.org/10.5281/zenodo.815721

- Fixed bug where using :func:`~mizani.breaks.log_breaks` with
  Numpy 1.13.0 led to a ``ValueError``.


v0.3.0
------
*(2017-04-24)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.557096.svg
   :target: https://doi.org/10.5281/zenodo.557096

- Added :func:`~mizani.palettes.xkcd_palette`, a palette that
  selects from 954 named colors.

- Added :func:`~mizani.palettes.crayon_palette`, a palette that
  selects from 163 named colors.

- Added :func:`cubehelix_pal`, a function that creates a continuous
  palette from the cubehelix system.

- Fixed bug where a color palette would raise an exception when
  passed a single scalar value instead of a list-like.

- :func:`~mizani.breaks.extended_breaks` and
  :func:`~mizani.breaks.mpl_breaks` now return a single break if
  the limits are equal. Previous, one run into an *Overflow* and
  the other returned a sequence filled with *n* of the same limit.

API Changes
***********

- :func:`~mizani.breaks.mpl_breaks` now returns a function
  that (strictly) expects a tuple with the minimum and maximum values.


v0.2.0
------
*(2017-01-27)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.260331.svg
   :target: https://doi.org/10.5281/zenodo.260331

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
