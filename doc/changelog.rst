Changelog
=========

v0.13.0
-------
*2024-10-24*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13986997.svg
  :target: https://doi.org/10.5281/zenodo.13986997

API Changes
***********

- Support for numpy `timedelta64` has been removed. It was not well supported
  in the first place, so removing it should be of consequence.

- `mizani.transforms.trans_new` function has been deprecated.

Enhancements
************

- `~mizani.breaks.breaks_date` has been slightly improved for the case when it
  generates monthly breaks.

New
***

- :class:`~mizani.transforms.trans` gained new method `diff_type_to_num` that
  should be helpful with some arithmetic operations for non-numeric domains.

v0.12.2
-------
*2024-09-04*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13688455.svg
  :target: https://doi.org/10.5281/zenodo.13688455

Bug Fixes
*********

- Fixed :class:`~mizani.bounds.squish` and
  :class:`~mizani.bounds.squish_infinite` to work for non writeable pandas
  series. This is broken for numpy 2.1.0.


v0.12.1
-------
*2024-08-19*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13346177.svg
  :target: https://doi.org/10.5281/zenodo.13346177

Enhancements
************
- Renamed "husl" color palette type to "hsluv". "husl" is the old name but
  we still work although not part of the API.

v0.12.0
-------
*2024-07-30*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13143647.svg
  :target: https://doi.org/10.5281/zenodo.13143647

API Changes
***********

- mizani now requires python 3.9 and above.

Bug Fixes
*********

- Fixed bug where a date with a timezone could lose the timezone. :issue:`45`.


v0.11.4
-------
*2024-05-24*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11281421.svg
  :target: https://doi.org/10.5281/zenodo.11281421

Bug Fixes
---------

- Fixed :class:`~mizani.bounds.squish` and
  :class:`~mizani.bounds.squish_infinite` so that they do not reuse
  numpy arrays. The users object is not modified.

  This also prevents exceptions where the numpy array backs a pandas
  object and it is protected by copy-on-write.


v0.11.3
-------
*2024-05-09*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11167016.svg
  :target: https://doi.org/10.5281/zenodo.11167016

Bug Fixes
---------

- Fixed bug when calculating monthly breaks where when the limits are narrow
  and do not align with the start and end of the month, there were no
  dates returned. (:issue:`42`)


v0.11.2
-------
*2024-04-26*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11074548.svg
  :target: https://doi.org/10.5281/zenodo.11074548

Bug Fixes
---------

- Added the ability to create reversed colormap for
  :class:`~mizani.palettes.cmap_pal` and
  :class:`~mizani.palettes.cmap_d_pal` using the matplotlib convention
  of `name_r`.


v0.11.1
-------
*2024-03-27*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10893760.svg
  :target: https://doi.org/10.5281/zenodo.10893760

Bug Fixes
---------

- Fix :class:`mizani.palettes.brewer_pal` to return exact colors in the when
  the requested colors are less than or equal to those in the palette.

- Add all matplotlib colormap and make them avalaible from
  :class:`~mizani.palettes.cmap_pal` and
  :class:`~mizani.palettes.cmap_d_pal` (:issue:`39`).

New
---

- Added :class:`~mizani.breaks.breaks_symlog` to calculate
  breaks for the symmetric logarithm transformation.

Changes
-------
- The default `big_mark` for :class:`~mizani.labels.label_number`
  has been changed from a comma to nothing.



v0.11.0
-------
*2024-02-12*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10650549.svg
  :target: https://doi.org/10.5281/zenodo.10650549

Enhancements
------------

- Removed FutureWarnings when using pandas 2.1.0

New
---

- Added :class:`~mizani.breaks.breaks_symlog` to calculate
  breaks for the symmetric logarithm transformation.

Changes
-------
- The default `big_mark` for :class:`~mizani.labels.label_number`
  has been changed from a comma to nothing.


v0.10.0
-------
*2023-07-28*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8193394.svg
   :target: https://doi.org/10.5281/zenodo.8193394

API Changes
***********

- :class:`~mizani.formatters.mpl_format` has been removed,
  :class:`~mizani.formatters.number_format` takes its place.

- :class:`~mizani.breaks.mpl_breaks` has been removed,
  :class:`~mizani.breaks.extended_breaks` has always been the default
  and it is sufficient.

- matplotlib has been removed as a dependency of mizani.

- mizani now requires python 3.9 and above.

- The units parameter for of :class:`~mizani.formatters.timedelta_format`
  now accepts the values `"min", "day", "week", "month"`,
  instead of `"m", "d", "w", "M"`.

- The naming convention for break formatting methods has changed from
  `*_format` to `label_*`. Specifically these methods have been renamed.

  * `comma_format` is now :class:`~mizani.formatters.label_comma`
  * `custom_format` is now :class:`~mizani.formatters.label_custom`
  * `currency_format` is now :class:`~mizani.formatters.label_currency`
  * `label_dollar` is now :class:`~mizani.formatters.label_dollar`
  * `percent_format` is now :class:`~mizani.formatters.label_percent`
  * `scientific_format` is now :class:`~mizani.formatters.label_scientific`
  * `date_format` is now :class:`~mizani.formatters.label_date`
  * `number_format` is now :class:`~mizani.formatters.label_number`
  * `log_format` is now :class:`~mizani.formatters.label_log`
  * `timedelta_format` is now :class:`~mizani.formatters.label_timedelta`
  * `pvalue_format` is now :class:`~mizani.formatters.label_pvalue`
  * `ordinal_format` is now :class:`~mizani.formatters.label_ordinal`
  * `number_bytes_format` is now :class:`~mizani.formatters.label_bytes`

- The naming convention for break calculating methods has changed from
  `*_breaks` to `breaks_*`. Specifically these methods have been renamed.

  * `log_breaks` is now :class:`~mizani.breaks.breaks_log`
  * `trans_minor_breaks` is now :class:`~mizani.breaks.minor_breaks_trans`
  * `date_breaks` is now :class:`~mizani.breaks. breaks_date`
  * `timedelta_breaks` is now :class:`~mizani.breaks. breaks_timedelta`
  * `extended_breaks` is now :class:`~mizani.breaks. breaks_extended`

- :class:`~mizani.transforms.trans.dataspace_is_numerical` has changed
  to :class:`~mizani.transforms.trans.domain_is_numerical` and it is now
  determined dynamically.

- The default `minor_breaks` for all transforms that are not linear
  are now calculated in dataspace. But only if the dataspace is
  numerical.

New
***
- :class:`~mizani.transforms.symlog_trans` for symmetric log transformation

v0.9.2
------

*2023-05-25*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7971866.svg
   :target: https://doi.org/10.5281/zenodo.7971866

Bug Fixes
*********

- Fixed regression in but in :class:`~mizani.formatters.date_format` where
  it cannot deal with UTC timezone from :class:`~datetime.timezone`
  :issue:`30`.

v0.9.1
------

*2023-05-19*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7951172.svg
   :target: https://doi.org/10.5281/zenodo.7951172

Bug Fixes
*********

- Fixed but in :class:`~mizani.formatters.date_format` to handle datetime
  sequences within the same timezone but a mixed daylight saving state.
  `(plotnine #687) <https://github.com/has2k1/plotnine/issues/687>`_

v0.9.0
------

*2023-04-15*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7951171.svg
   :target: https://doi.org/10.5281/zenodo.7951171

API Changes
************

- `palettable` dropped as a dependency.

Bug Fixes
*********

- Fixed bug in :class:`~mizani.transforms.datetime_trans` where
  a pandas series with an index that did not start at 0 could not
  be transformed.

- Install tzdata on pyiodide/emscripten. :issue:`27`

v0.8.1
------

*2022-09-28*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7120121.svg
   :target: https://doi.org/10.5281/zenodo.7120121

Bug Fixes
*********

- Fixed regression bug in :class:`~mizani.formatters.log_format` for
  where formatting for bases 2, 8 and 16 would fail if the values were
  float-integers.

Enhancements
************
- :class:`~mizani.formatters.log_format` now uses exponent notation
  for bases other than base 10.

v0.8.0
------

*2022-09-26*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7113103.svg
   :target: https://doi.org/10.5281/zenodo.7113103

API Changes
***********

- The ``lut`` parameter of :class:`~mizani.palettes.cmap_pal` and
  :class:`~mizani.palettes.cmap_d_pal` has been deprecated and will
  removed in a future version.

- :class:`~mizani.transforms.datetime_trans` gained parameter ``tz``
  that controls the timezone of the transformation.

- :class:`~mizani.formatters.log_format` gained boolean parameter
  ``mathtex`` for TeX values as understood matplotlib instead of
  values in scientific notation.

Bug Fixes
*********

- Fixed bug in :class:`~mizani.bounds.zero_range` where ``uint64``
  values would cause a RuntimeError.

v0.7.4
------
*2022-04-02*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6408007.svg
   :target: https://doi.org/10.5281/zenodo.6408007

API Changes
***********

- :class:`~mizani.formatters.comma_format` is now imported
  automatically when using ``*``.

- Fixed issue with :class:`~mizani.scales.scale_discrete` so that if
  you train on data with ``Nan`` and specify and old range that also
  has ``NaN``, the result range does not include two ``NaN`` values.

v0.7.3
------
*(2020-10-29)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4633364.svg
   :target: https://doi.org/10.5281/zenodo.4633364


Bug Fixes
*********
- Fixed log_breaks for narrow range if base=2 (:issue:`76`).


v0.7.2
------
*(2020-10-29)*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4633357.svg
   :target: https://doi.org/10.5281/zenodo.4633357

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
