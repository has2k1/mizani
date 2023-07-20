from __future__ import annotations

import typing
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import dateutil.rrule as rr

if typing.TYPE_CHECKING:
    from mizani.typing import (
        TzInfo,
    )


class DateFrequency(IntEnum):
    """
    Date Frequency

    Matching the dateutils constants
    """

    __order__ = "YEARLY MONTHLY DAILY HOURLY MINUTELY SECONDLY MICROSECONDLY"
    YEARLY = rr.YEARLY
    MONTHLY = rr.MONTHLY
    DAILY = rr.DAILY
    HOURLY = rr.HOURLY
    MINUTELY = rr.MINUTELY
    SECONDLY = rr.SECONDLY
    MICROSECONDLY = SECONDLY + 1


DF = DateFrequency


@dataclass
class date_breaks_info:
    """
    Information required to generate sequence of date breaks
    """

    frequency: DateFrequency
    n: int
    width: int
    start: datetime
    until: datetime
    tz: TzInfo | None
