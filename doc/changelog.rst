Changelog
=========

v0.7.2
------
*(2020-10-29)*

Bug Fixes
*********
- Fixed bug in :func:`~mizani.bounds.rescale_max` to properly handle
  values whose maximum is zero (:issue:`16`).

v0.7.1
------
*(2020-06-05)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3877546.svg
   :target: https://doi.org/10.5281/zenodo.3877546

Bug Fixes
*********
- Fixed regression in :func:`mizani.scales.scale_discrete.train` when
  trainning on values with some categoricals that have common elements.

v0.7.0
------
*(2020-06-04)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3876327.svg
   :target: https://doi.org/10.5281/zenodo.3876327

Bug Fixes
*********
- Fixed issue with :class:`mizani.formatters.log_breaks` where non-linear
  breaks could not be generated if the limits where greater than the
  largest integer ``sys.maxsize``.

- Fixed :func:`mizani.palettes.gradient_n_pal` to return ``nan`` for
  ``nan`` values.

- Fixed :func:`mizani.scales.scale_discrete.train` when training categoricals
  to maintain the order.
  `(plotnine #381) <https://github.com/has2k1/plotnine/issues/381>`_

v0.6.0
------
*(2019-08-15)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3369191.svg
   :target: https://doi.org/10.5281/zenodo.3369191

New
***
- Added :class:`~mizani.formatters.pvalue_format`
- Added :class:`~mizani.formatters.ordinal_format`
- Added :class:`~mizani.formatters.number_bytes_format`
- Added :func:`~mizani.transforms.pseudo_log_trans`
- Added :class:`~mizani.transforms.reciprocal_trans`
- Added :func:`~mizani.transforms.modulus_trans`

Enhancements
************
- :class:`mizani.breaks.date_breaks` now supports intervals in the
   order of seconds.

- :class:`mizani.palettes.brewer_pal` now supports a direction argument
  to control the order of the returned colors.

API Changes
***********
- :func:`~mizani.transforms.boxcox_trans` now only accepts positive
  values. For both positive and negative values,
  :func:`~mizani.transforms.modulus_trans` has been added.

v0.5.4
------
*(2019-03-26)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.62319878.svg
   :target: https://doi.org/10.5281/zenodo.62319878

Enhancements
************
- :class:`mizani.formatters.log_format` now does a better job of
  approximating labels for numbers like ``3.000000000000001e-05``.

API Changes
-----------

- ``exponent_threshold`` parameter of :class:`mizani.formatters.log_format` has
  been deprecated.

v0.5.3
------
*(2018-12-24)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2526010.svg
   :target: https://doi.org/10.5281/zenodo.2526010


API Changes
-----------
- Log transforms now default to ``base - 2`` minor breaks.
  So base 10 has 8 minor breaks and 9 partitions,
  base 8 has 6 minor breaks and 7 partitions, ...,
  base 2 has 0 minor breaks and a single partition.


v0.5.2
------
*(2018-10-17)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2525577.svg
   :target: https://doi.org/10.5281/zenodo.2525577

Bug Fixes
*********

- Fixed issue where some functions that took pandas series
  would return output where the index did not match that of the input.

v0.5.1
------
*(2018-10-15)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1464266.svg
   :target: https://doi.org/10.5281/zenodo.1464266

Bug Fixes
*********

- Fixed issue with :class:`~mizani.breaks.log_breaks`, so that it does
  not fail needlessly when the limits in the (0, 1) range.

Enhancements
************

- Changed :class:`~mizani.formatters.log_format` to return better
  formatted breaks.

v0.5.0
------
*(2018-11-10)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1461621.svg
   :target: https://doi.org/10.5281/zenodo.1461621

API Changes
***********

- Support for python 2 has been removed.

- :meth:`~mizani.breaks.minor_breaks.call` and
   meth:`~mizani.breaks.trans_minor_breaks.call` now accept optional
   parameter ``n`` which is the number of minor breaks between any two
   major breaks.

- The parameter `nan_value` has be renamed to `na_value`.

- The parameter `nan_rm` has be renamed to `na_rm`.

Enhancements
************

- Better support for handling missing values when training discrete
  scales.

- Changed the algorithm for :class:`~mizani.breaks.log_breaks`, it can
  now return breaks that do not fall on the integer powers of the base.

v0.4.6
------
*(2018-03-20)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1204282.svg
   :target: https://doi.org/10.5281/zenodo.1204282

- Added :class:`~mizani.bounds.squish`

v0.4.5
------
*(2018-03-09)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1204222.svg
   :target: https://doi.org/10.5281/zenodo.1204222

- Added :class:`~mizani.palettes.identity_pal`
- Added :class:`~mizani.palettes.cmap_d_pal`

v0.4.4
------
*(2017-12-13)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1115676.svg
   :target: https://doi.org/10.5281/zenodo.1115676

- Fixed :class:`~mizani.formatters.date_format` to respect the timezones
  of the dates (:issue:`8`).

v0.4.3
------
*(2017-12-01)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1069571.svg
   :target: https://doi.org/10.5281/zenodo.1069571

- Changed :class:`~mizani.breaks.date_breaks` to have more variety
  in the spacing between the breaks.

- Fixed :class:`~mizani.formatters.date_format` to respect time part
  of the date (:issue:`7`).

v0.4.2
------
*(2017-11-06)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1042715.svg
   :target: https://doi.org/10.5281/zenodo.1042715

- Fixed (regression) break calculation for the non ordinal transforms.


v0.4.1
------
*(2017-11-04)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1041981.svg
   :target: https://doi.org/10.5281/zenodo.1041981

- :class:`~mizani.transforms.trans` objects can now be instantiated
  with parameter to override attributes of the instance. And the
  default methods for computing breaks and minor breaks on the
  transform instance are not class attributes, so they can be
  modified without global repercussions.

v0.4.0
------
*(2017-10-24)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1035809.svg
   :target: https://doi.org/10.5281/zenodo.1035809

API Changes
***********
- Breaks and formatter generating functions have been converted to
  classes, with a ``__call__`` method. How they are used has not
  changed, but this makes them move flexible.

- ``ExtendedWilkson`` class has been removed.
  :func:`~mizani.breaks.extended_breaks` now contains the implementation
  of the break calculating algorithm.


v0.3.4
------
*(2017-09-12)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.890135.svg
   :target: https://doi.org/10.5281/zenodo.890135

- Fixed issue where some formatters methods failed if passed empty
  ``breaks`` argument.

- Fixed issue with :func:`~mizani.breaks.log_breaks` where if the
  limits were with in the same order of magnitude the calculated
  breaks were always the ends of the order of magnitude.

  Now :python:`log_breaks()((35, 50))` returns ``[35,  40,  45,  50]``
  as breaks instead of ``[1, 100]``.


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
