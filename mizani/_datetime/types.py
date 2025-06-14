from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


class DateTimeRounder(Protocol):
    """
    Round a datetime object
    """

    @classmethod
    def floor(cls, d: datetime) -> datetime: ...

    @classmethod
    def ceil(cls, d: datetime) -> datetime: ...
