"""Calendar adapters: a ``CalendarAdapter`` protocol with sealed result types, and its implementations."""

from booking_truth.calendars.base import (
    BookingRecord,
    CalendarAdapter,
    ListResult,
    NotFound,
    ReadResult,
    Slot,
    Slots,
    SlotsResult,
    Unavailable,
    WriteOk,
    WriteRejected,
    WriteResult,
    WriteUnknown,
)
from booking_truth.calendars.calcom import CalcomAdapter
from booking_truth.calendars.factory import CalendarOptions, build_calendar, calendar_options

__all__ = [
    "BookingRecord",
    "CalcomAdapter",
    "CalendarAdapter",
    "CalendarOptions",
    "ListResult",
    "NotFound",
    "ReadResult",
    "Slot",
    "Slots",
    "SlotsResult",
    "Unavailable",
    "WriteOk",
    "WriteRejected",
    "WriteResult",
    "WriteUnknown",
    "build_calendar",
    "calendar_options",
]
