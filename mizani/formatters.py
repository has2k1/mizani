"""
This module has been deprecated in favour of :mod:`~mizani.labels`
"""

from .labels import (
    label_bytes,
    label_comma,
    label_currency,
    label_custom,
    label_date,
    label_dollar,
    label_log,
    label_number,
    label_ordinal,
    label_percent,
    label_pvalue,
    label_scientific,
    label_timedelta,
)

# Deprecated
comma_format = label_comma
custom_format = label_custom
currency_format = label_currency
label_format = label_dollar
percent_format = label_percent
scientific_format = label_scientific
date_format = label_date
number_format = label_number
log_format = label_log
timedelta_format = label_timedelta
pvalue_format = label_pvalue
ordinal_format = label_ordinal
number_bytes_format = label_bytes
