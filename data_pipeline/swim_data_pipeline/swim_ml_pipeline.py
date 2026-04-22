"""
FAA SWIM SCDS — Continuous Flight Delay Feature Pipeline
=========================================================
Connects to the R14 Flight Data queue, parses fltdMessage XML,
extracts delay features, maintains tail-number rotation state,
enriches with METAR weather, applies LADD filtering, and writes
records to a rolling Parquet file ready for ST-GCN / ML input.

ENVIRONMENT VARIABLES
---------------------
    SCDS_HOST         tcps://ems2.swim.faa.gov:55443
    SCDS_VPN          TFMS
    SCDS_USERNAME     your_username
    SCDS_PASSWORD     your_password
    SCDS_QUEUE_FLIGHT your_username.TFMS.UUID.OUT
    LADD_PATH         /path/to/IndustryLADD.csv
    OUTPUT_DIR        /path/to/output/directory  (default: ./pipeline_output)

USAGE
-----
    # Run continuously, flushing to parquet every 500 records
    python swim_pipeline.py

    # Run for 1 hour then exit
    python swim_pipeline.py --duration 3600

    # Dry run — print records to stdout, no file output
    python swim_pipeline.py --dry-run
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Solace
# ---------------------------------------------------------------------------
try:
    from solace.messaging.messaging_service import MessagingService
    from solace.messaging.resources.queue import Queue
    from solace.messaging.receiver.persistent_message_receiver import PersistentMessageReceiver
    from solace.messaging.receiver.message_receiver import MessageHandler
except ImportError:
    sys.exit("Run: pip install solace-pubsubplus")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("swim_pipeline")

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
CREDS = {
    "host":     os.environ.get("SCDS_HOST",         "tcps://ems2.swim.faa.gov:55443"),
    "vpn":      os.environ.get("SCDS_VPN",           "TFMS"),
    "username": os.environ.get("SCDS_USERNAME",      "YOUR_USERNAME"),
    "password": os.environ.get("SCDS_PASSWORD",      "YOUR_PASSWORD"),
    "queue":    os.environ.get("SCDS_QUEUE_FLIGHT",  "YOUR_USERNAME.TFMS.UUID.OUT"),
}

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./pipeline_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LADD filter
# ---------------------------------------------------------------------------
_LADD_PATH = os.environ.get("LADD_PATH", "IndustryLADD.csv")
try:
    LADD_SET = set(
        pd.read_csv(_LADD_PATH)["registration"].str.upper().str.strip()
    )
    log.info("LADD filter loaded: %d registrations", len(LADD_SET))
except FileNotFoundError:
    log.warning("LADD file not found at '%s' — no filtering applied.", _LADD_PATH)
    LADD_SET = set()
except Exception as e:
    log.warning("LADD load error: %s", e)
    LADD_SET = set()

# ---------------------------------------------------------------------------
# Airport timezone lookup (ICAO -> pytz timezone string)
# Covers all major US airports — extend as needed
# ---------------------------------------------------------------------------
AIRPORT_TZ = {
    "KATL": "America/New_York",   "KORD": "America/Chicago",
    "KLAX": "America/Los_Angeles","KDFW": "America/Chicago",
    "KDEN": "America/Denver",     "KJFK": "America/New_York",
    "KSFO": "America/Los_Angeles","KLAS": "America/Los_Angeles",
    "KSEA": "America/Los_Angeles","KMCO": "America/New_York",
    "KEWR": "America/New_York",   "KBOS": "America/New_York",
    "KMSP": "America/Chicago",    "KPHX": "America/Phoenix",
    "KDTW": "America/Detroit",    "KPHL": "America/New_York",
    "KLGA": "America/New_York",   "KIAH": "America/Chicago",
    "KBWI": "America/New_York",   "KFLL": "America/New_York",
    "KSLC": "America/Denver",     "KDCA": "America/New_York",
    "KMDW": "America/Chicago",    "KSDF": "America/Kentucky/Louisville",
    "KMEM": "America/Chicago",    "KCLE": "America/New_York",
    "KPVD": "America/New_York",   "KCLT": "America/New_York",
    "KRDU": "America/New_York",   "KONT": "America/Los_Angeles",
    "KMDT": "America/New_York",   "KBWI": "America/New_York",
}

def get_local_time(utc_dt: Optional[datetime], icao: str) -> Optional[datetime]:
    """Convert UTC datetime to local time for an airport."""
    if utc_dt is None:
        return None
    try:
        import pytz
        tz_str = AIRPORT_TZ.get(icao.upper(), "America/New_York")
        tz = pytz.timezone(tz_str)
        return utc_dt.astimezone(tz)
    except Exception:
        return utc_dt

# ---------------------------------------------------------------------------
# METAR weather cache
# Polls NOAA Aviation Weather Center API, caches per airport with 10-min TTL
# ---------------------------------------------------------------------------
class METARCache:
    def __init__(self, ttl_seconds: int = 600):
        self._cache: dict = {}
        self._ttl = ttl_seconds

    def get(self, icao: str) -> dict:
        icao = icao.upper()
        entry = self._cache.get(icao)
        if entry:
            age = (datetime.now(timezone.utc) - entry["ts"]).total_seconds()
            if age < self._ttl:
                return entry["data"]
        data = self._fetch(icao)
        self._cache[icao] = {"ts": datetime.now(timezone.utc), "data": data}
        return data

    def _fetch(self, icao: str) -> dict:
        try:
            url = "https://aviationweather.gov/api/data/metar"
            resp = requests.get(
                url,
                params={"ids": icao, "format": "json", "hours": 1},
                timeout=5
            )
            obs = resp.json()
            if not obs:
                return {}
            m = obs[0]
            wspd = m.get("wspd") or 0
            return {
                "temp_c":          m.get("temp"),
                "wind_speed_m_s":  round(float(wspd) * 0.514444, 2),
                "wind_dir_deg":    m.get("wdir"),
                "ceiling_height_m": _parse_ceiling(m.get("cover", "")),
                "visibility_m":    m.get("visib"),
            }
        except Exception as e:
            log.debug("METAR fetch failed for %s: %s", icao, e)
            return {}

def _parse_ceiling(cover_str: str) -> Optional[float]:
    """Extract ceiling height in meters from METAR cover string."""
    if not cover_str:
        return None
    try:
        # cover_str like "BKN050" or "OVC010"
        import re
        m = re.search(r"(\d{3})$", cover_str.strip())
        if m:
            hundreds_ft = int(m.group(1))
            return round(hundreds_ft * 100 * 0.3048, 1)
    except Exception:
        pass
    return None

METAR_CACHE = METARCache(ttl_seconds=600)

# ---------------------------------------------------------------------------
# Tail number state store (in-memory rotation chain tracker)
# ---------------------------------------------------------------------------
@dataclass
class TailState:
    flight_id:       str   = ""
    origin:          str   = ""
    dest:            str   = ""
    arr_ts_actual:   Optional[datetime] = None
    dep_ts_actual:   Optional[datetime] = None
    arr_delay:       Optional[float]    = None
    dep_delay:       Optional[float]    = None
    leg_number_day:  int   = 0
    cum_dep_delay:   float = 0.0
    cum_arr_delay:   float = 0.0
    flight_date:     str   = ""
    updated_at:      Optional[datetime] = None

class TailStateStore:
    def __init__(self):
        self._store: dict[str, TailState] = {}

    def get(self, tail: str) -> Optional[TailState]:
        return self._store.get(tail.upper())

    def update(self, tail: str, state: TailState):
        self._store[tail.upper()] = state

    def compute_features(
        self,
        tail: str,
        flight_id: str,
        origin: str,
        dest: str,
        dep_ts_actual: Optional[datetime],
        arr_ts_actual: Optional[datetime],
        dep_delay: Optional[float],
        arr_delay: Optional[float],
        flight_date: str,
    ) -> dict:
        """
        Given incoming flight data, compute rotation chain features
        from the previous state, then update state for next flight.
        """
        prev = self.get(tail)

        # --- Rotation linkage features ---
        prev_flight_id      = prev.flight_id      if prev else None
        prev_origin         = prev.origin          if prev else None
        prev_dest           = prev.dest            if prev else None
        prev_arr_ts_actual  = prev.arr_ts_actual   if prev else None
        prev_arr_delay      = prev.arr_delay       if prev else None
        prev_dep_delay      = prev.dep_delay       if prev else None

        # Turnaround = current dep - previous arr
        turnaround_minutes = None
        if dep_ts_actual and prev_arr_ts_actual:
            diff = (dep_ts_actual - prev_arr_ts_actual).total_seconds() / 60.0
            if 0 <= diff <= 720:  # sanity: 0-12 hours
                turnaround_minutes = round(diff, 1)

        # Rotation continuity: previous dest == current origin
        rotation_continuity = (
            1 if prev_dest and origin and prev_dest.upper() == origin.upper()
            else 0
        )

        # Reset leg count if new day
        if prev and prev.flight_date == flight_date:
            leg_number_day = prev.leg_number_day + 1
            cum_dep_delay  = (prev.cum_dep_delay or 0) + (dep_delay or 0)
            cum_arr_delay  = (prev.cum_arr_delay or 0) + (arr_delay or 0)
        else:
            leg_number_day = 1
            cum_dep_delay  = dep_delay or 0
            cum_arr_delay  = arr_delay or 0

        # --- Update state ---
        new_state = TailState(
            flight_id      = flight_id,
            origin         = origin,
            dest           = dest,
            arr_ts_actual  = arr_ts_actual,
            dep_ts_actual  = dep_ts_actual,
            arr_delay      = arr_delay,
            dep_delay      = dep_delay,
            leg_number_day = leg_number_day,
            cum_dep_delay  = cum_dep_delay,
            cum_arr_delay  = cum_arr_delay,
            flight_date    = flight_date,
            updated_at     = datetime.now(timezone.utc),
        )
        self.update(tail, new_state)

        return {
            "prev_flight_id_same_tail":  prev_flight_id,
            "prev_origin":               prev_origin,
            "prev_dest":                 prev_dest,
            "prev_arr_ts_actual_utc":    prev_arr_ts_actual.isoformat() if prev_arr_ts_actual else None,
            "prev_arr_delay":            prev_arr_delay,
            "prev_dep_delay":            prev_dep_delay,
            "turnaround_minutes":        turnaround_minutes,
            "rotation_continuity_flag":  rotation_continuity,
            "aircraft_leg_number_day":   leg_number_day,
            "cum_dep_delay_aircraft_day": cum_dep_delay,
            "cum_arr_delay_aircraft_day": cum_arr_delay,
        }

TAIL_STORE = TailStateStore()

# ---------------------------------------------------------------------------
# Flight feature record — maps directly to your ML pipeline feature list
# ---------------------------------------------------------------------------
@dataclass
class FlightFeatureRecord:
    # Identity
    sampled_at:               str            = ""
    flight_id:                str            = ""   # gufi
    tail_number:              str            = ""   # acid (callsign used as proxy)
    reporting_airline:        str            = ""
    origin:                   str            = ""
    dest:                     str            = ""
    route_key:                str            = ""
    flight_date:              str            = ""

    # Local time features
    dep_hour_local:           Optional[int]  = None
    dep_weekday_local:        Optional[int]  = None
    dep_month_local:          Optional[int]  = None

    # UTC timestamps
    dep_ts_sched_utc:         str            = ""
    dep_ts_actual_utc:        str            = ""
    arr_ts_sched_utc:         str            = ""
    arr_ts_actual_utc:        str            = ""

    # Weather — departure airport
    dep_temp_c:               Optional[float] = None
    dep_wind_speed_m_s:       Optional[float] = None
    dep_wind_dir_deg:         Optional[float] = None
    dep_ceiling_height_m:     Optional[float] = None

    # Weather — arrival airport
    arr_temp_c:               Optional[float] = None
    arr_wind_speed_m_s:       Optional[float] = None
    arr_wind_dir_deg:         Optional[float] = None
    arr_ceiling_height_m:     Optional[float] = None

    # Delays (labels)
    dep_delay:                Optional[float] = None
    dep_del15:                Optional[int]   = None
    arr_delay:                Optional[float] = None
    arr_del15:                Optional[int]   = None

    # Rotation chain features
    prev_flight_id_same_tail: Optional[str]  = None
    prev_origin:              Optional[str]  = None
    prev_dest:                Optional[str]  = None
    prev_arr_ts_actual_utc:   Optional[str]  = None
    prev_arr_delay:           Optional[float] = None
    prev_dep_delay:           Optional[float] = None
    turnaround_minutes:       Optional[float] = None
    rotation_continuity_flag: Optional[int]  = None
    aircraft_leg_number_day:  Optional[int]  = None
    cum_dep_delay_aircraft_day: Optional[float] = None
    cum_arr_delay_aircraft_day: Optional[float] = None

    # Status
    is_cancelled:             int            = 0
    is_diverted:              int            = 0

    # Schedule
    crs_elapsed_time:         Optional[float] = None
    dep_time_blk:             str            = ""
    arr_time_blk:             str            = ""

    # Message metadata
    msg_type:                 str            = ""
    source_facility:          str            = ""
    source_timestamp:         str            = ""

# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag

def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _delay_minutes(
    scheduled: Optional[datetime],
    actual: Optional[datetime]
) -> Optional[float]:
    if scheduled and actual:
        return round((actual - scheduled).total_seconds() / 60.0, 1)
    return None

def _time_block(hour: Optional[int]) -> str:
    """Map departure hour to FAA time block label."""
    if hour is None:
        return ""
    blocks = [
        (0,  6,  "0001-0559"),
        (6,  7,  "0600-0659"),
        (7,  8,  "0700-0759"),
        (8,  9,  "0800-0859"),
        (9,  10, "0900-0959"),
        (10, 11, "1000-1059"),
        (11, 12, "1100-1159"),
        (12, 13, "1200-1259"),
        (13, 14, "1300-1359"),
        (14, 15, "1400-1459"),
        (15, 16, "1500-1559"),
        (16, 17, "1600-1659"),
        (17, 18, "1700-1759"),
        (18, 19, "1800-1859"),
        (19, 20, "1900-1959"),
        (20, 21, "2000-2059"),
        (21, 22, "2100-2159"),
        (22, 23, "2200-2259"),
        (23, 24, "2300-2359"),
    ]
    for start, end, label in blocks:
        if start <= hour < end:
            return label
    return ""

def _find_in(element: ET.Element, local_tag: str) -> Optional[str]:
    """Search subtree for first element with this local tag, return text."""
    for elem in element.iter():
        if _strip_ns(elem.tag) == local_tag:
            return elem.text.strip() if elem.text else None
    return None

def _find_attr_in(
    element: ET.Element,
    local_tag: str,
    attr: str
) -> Optional[str]:
    """Search subtree for first element with local tag, return attribute value."""
    for elem in element.iter():
        if _strip_ns(elem.tag) == local_tag:
            # Try plain attr name and strip-ns version
            val = elem.get(attr)
            if val:
                return val.strip()
            for k, v in elem.attrib.items():
                if _strip_ns(k) == attr:
                    return v.strip()
    return None

def parse_fltd_output(xml_text: str) -> list[FlightFeatureRecord]:
    """
    Parse a full fltdOutput message which may contain multiple fltdMessage
    elements. Returns one FlightFeatureRecord per relevant fltdMessage.

    Based on actual observed schema:
      <ns0:tfmDataService>
        <ns0:fltdOutput>
          <ns2:fltdMessage acid="..." airline="..." arrArpt="..." depArpt="..."
                           flightRef="..." msgType="..." sourceTimeStamp="...">
            <ns2:departureInformation>
              <ns3:ncsmFlightTimeData>
                <ns3:etd etdType="ACTUAL" timeValue="..." />
                <ns3:eta etaType="ESTIMATED" timeValue="..." />
              </ns3:ncsmFlightTimeData>
            </ns2:departureInformation>
            <ns4:gufi>...</ns4:gufi>
            <ns4:igtd>...</ns4:igtd>   ← scheduled departure
            <ns4:departurePoint><ns4:airport>KMEM</ns4:airport></ns4:departurePoint>
            <ns4:arrivalPoint><ns4:airport>KCLE</ns4:airport></ns4:arrivalPoint>
          </ns2:fltdMessage>
        </ns0:fltdOutput>
      </ns0:tfmDataService>
    """
    records = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.debug("XML parse error: %s", e)
        return records

    # Find all fltdMessage elements anywhere in the tree
    for elem in root.iter():
        if _strip_ns(elem.tag) != "fltdMessage":
            continue

        msg_type      = elem.get("msgType", "")
        acid          = elem.get("acid", "")
        airline       = elem.get("airline", "")
        dep_arpt      = elem.get("depArpt", "")
        arr_arpt      = elem.get("arrArpt", "")
        flight_ref    = elem.get("flightRef", "")
        source_fac    = elem.get("sourceFacility", "")
        source_ts_str = elem.get("sourceTimeStamp", "")

        # Only process departure and amendment messages that have time data
        # trackInformation messages don't have scheduled/actual times we need
        relevant_msg_types = {
            "departureInformation",
            "flightPlanAmendmentInformation",
            "flightPlanInformation",
            "arrivalInformation",
            "flightPlanFiledInformation",
        }
        if msg_type not in relevant_msg_types:
            continue

        # LADD check on callsign
        if acid.upper() in LADD_SET:
            continue

        # --- Extract times ---
        gufi     = _find_in(elem, "gufi") or flight_ref
        igtd_str = _find_in(elem, "igtd")  # scheduled gate departure

        # ETD — look for etdType ACTUAL first, then ESTIMATED
        etd_actual_str    = None
        etd_estimated_str = None
        eta_estimated_str = None
        eta_actual_str    = None

        for etd_elem in elem.iter():
            if _strip_ns(etd_elem.tag) == "etd":
                etd_type = etd_elem.get("etdType", "")
                tv       = etd_elem.get("timeValue", "")
                if etd_type == "ACTUAL":
                    etd_actual_str = tv
                elif etd_type == "ESTIMATED" and not etd_estimated_str:
                    etd_estimated_str = tv

        for eta_elem in elem.iter():
            if _strip_ns(eta_elem.tag) == "eta":
                eta_type = eta_elem.get("etaType", "")
                tv       = eta_elem.get("timeValue", "")
                if eta_type == "ACTUAL":
                    eta_actual_str = tv
                elif eta_type == "ESTIMATED" and not eta_estimated_str:
                    eta_estimated_str = tv

        # Scheduled times
        dep_sched_dt = _parse_iso(igtd_str)

        # Actual/estimated departure
        dep_actual_dt = _parse_iso(etd_actual_str or etd_estimated_str)

        # Arrival times — eta is estimated, look for actual too
        arr_sched_dt  = None  # SWIM R14 doesn't always provide sched arrival
        arr_actual_dt = _parse_iso(eta_actual_str or eta_estimated_str)

        # Skip if we don't have at least a scheduled departure
        if dep_sched_dt is None:
            continue

        # --- Compute delays ---
        dep_delay = _delay_minutes(dep_sched_dt, dep_actual_dt)
        arr_delay = None  # computed when arr_actual available
        dep_del15 = int(dep_delay >= 15) if dep_delay is not None else None
        arr_del15 = None

        # --- Local time features ---
        dep_local = get_local_time(dep_sched_dt, dep_arpt)
        dep_hour_local    = dep_local.hour     if dep_local else None
        dep_weekday_local = dep_local.isoweekday() if dep_local else None  # 1=Mon 7=Sun
        dep_month_local   = dep_local.month    if dep_local else None
        flight_date       = dep_local.strftime("%Y-%m-%d") if dep_local else \
                            dep_sched_dt.strftime("%Y-%m-%d")

        # --- Schedule block ---
        dep_time_blk = _time_block(dep_hour_local)
        arr_time_blk = ""  # filled in when arr data available

        # --- CRS elapsed time ---
        crs_elapsed = None
        if dep_sched_dt and arr_sched_dt:
            crs_elapsed = round(
                (arr_sched_dt - dep_sched_dt).total_seconds() / 60.0, 1
            )

        # --- Weather enrichment ---
        dep_wx = METAR_CACHE.get(dep_arpt) if dep_arpt else {}
        arr_wx = METAR_CACHE.get(arr_arpt) if arr_arpt else {}

        # --- Rotation chain features ---
        rotation = TAIL_STORE.compute_features(
            tail          = acid,
            flight_id     = gufi,
            origin        = dep_arpt,
            dest          = arr_arpt,
            dep_ts_actual = dep_actual_dt,
            arr_ts_actual = arr_actual_dt,
            dep_delay     = dep_delay,
            arr_delay     = arr_delay,
            flight_date   = flight_date,
        )

        # --- Build record ---
        rec = FlightFeatureRecord(
            sampled_at            = datetime.now(timezone.utc).isoformat(),
            flight_id             = gufi,
            tail_number           = acid,
            reporting_airline     = airline,
            origin                = dep_arpt,
            dest                  = arr_arpt,
            route_key             = f"{dep_arpt}_{arr_arpt}",
            flight_date           = flight_date,
            dep_hour_local        = dep_hour_local,
            dep_weekday_local     = dep_weekday_local,
            dep_month_local       = dep_month_local,
            dep_ts_sched_utc      = dep_sched_dt.isoformat() if dep_sched_dt else "",
            dep_ts_actual_utc     = dep_actual_dt.isoformat() if dep_actual_dt else "",
            arr_ts_sched_utc      = arr_sched_dt.isoformat() if arr_sched_dt else "",
            arr_ts_actual_utc     = arr_actual_dt.isoformat() if arr_actual_dt else "",
            dep_temp_c            = dep_wx.get("temp_c"),
            dep_wind_speed_m_s    = dep_wx.get("wind_speed_m_s"),
            dep_wind_dir_deg      = dep_wx.get("wind_dir_deg"),
            dep_ceiling_height_m  = dep_wx.get("ceiling_height_m"),
            arr_temp_c            = arr_wx.get("temp_c"),
            arr_wind_speed_m_s    = arr_wx.get("wind_speed_m_s"),
            arr_wind_dir_deg      = arr_wx.get("wind_dir_deg"),
            arr_ceiling_height_m  = arr_wx.get("ceiling_height_m"),
            dep_delay             = dep_delay,
            dep_del15             = dep_del15,
            arr_delay             = arr_delay,
            arr_del15             = arr_del15,
            prev_flight_id_same_tail  = rotation["prev_flight_id_same_tail"],
            prev_origin               = rotation["prev_origin"],
            prev_dest                 = rotation["prev_dest"],
            prev_arr_ts_actual_utc    = rotation["prev_arr_ts_actual_utc"],
            prev_arr_delay            = rotation["prev_arr_delay"],
            prev_dep_delay            = rotation["prev_dep_delay"],
            turnaround_minutes        = rotation["turnaround_minutes"],
            rotation_continuity_flag  = rotation["rotation_continuity_flag"],
            aircraft_leg_number_day   = rotation["aircraft_leg_number_day"],
            cum_dep_delay_aircraft_day = rotation["cum_dep_delay_aircraft_day"],
            cum_arr_delay_aircraft_day = rotation["cum_arr_delay_aircraft_day"],
            is_cancelled              = 0,
            is_diverted               = 0,
            crs_elapsed_time          = crs_elapsed,
            dep_time_blk              = dep_time_blk,
            arr_time_blk              = arr_time_blk,
            msg_type                  = msg_type,
            source_facility           = source_fac,
            source_timestamp          = source_ts_str,
        )

        records.append(rec)

    return records

# ---------------------------------------------------------------------------
# Pipeline message handler
# ---------------------------------------------------------------------------

class PipelineHandler(MessageHandler):

    def __init__(
        self,
        flush_every:  int  = 500,
        dry_run:      bool = False,
        max_records:  int  = 0,
    ):
        self.flush_every  = flush_every
        self.dry_run      = dry_run
        self.max_records  = max_records
        self.buffer:      list[FlightFeatureRecord] = []
        self.total_written = 0
        self.msg_count    = 0
        self.skip_count   = 0
        self.error_count  = 0
        self._done        = False
        self._file_index  = 0

    @property
    def done(self) -> bool:
        return self._done

    def on_message(self, message):
        if self._done:
            return
        self.msg_count += 1

        try:
            payload = message.get_payload_as_string()
        except Exception:
            self.skip_count += 1
            return

        if not payload:
            self.skip_count += 1
            return

        records = parse_fltd_output(payload)

        if not records:
            self.skip_count += 1
            return

        for rec in records:
            if self.dry_run:
                d = {k: v for k, v in asdict(rec).items() if v not in (None, "")}
                print(json.dumps(d, indent=2))
            else:
                self.buffer.append(rec)

        self.total_written += len(records)

        # Progress log every 100 messages
        if self.msg_count % 100 == 0:
            log.info(
                "msgs: %d  records: %d  skipped: %d  errors: %d  buffer: %d",
                self.msg_count, self.total_written,
                self.skip_count, self.error_count, len(self.buffer)
            )

        # Flush buffer to parquet
        if not self.dry_run and len(self.buffer) >= self.flush_every:
            self._flush()

        # Stop if max_records reached
        if self.max_records and self.total_written >= self.max_records:
            if not self.dry_run:
                self._flush()
            log.info("Reached max_records=%d — stopping.", self.max_records)
            self._done = True

    def _flush(self):
        if not self.buffer:
            return
        self._file_index += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = OUTPUT_DIR / f"swim_flights_{ts}_{self._file_index:04d}.parquet"
        df = pd.DataFrame([asdict(r) for r in self.buffer])
        df.to_parquet(path, index=False)
        log.info("Flushed %d records to %s", len(self.buffer), path)
        self.buffer.clear()

    def flush_remaining(self):
        if self.buffer:
            self._flush()

# ---------------------------------------------------------------------------
# TLS helper
# ---------------------------------------------------------------------------

def _get_trust_store() -> str:
    override = os.environ.get("SOLACE_TRUST_STORE")
    if override and os.path.exists(override):
        return override
    try:
        import certifi
        return certifi.where()
    except ImportError:
        pass
    for path in (
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/cert.pem",
    ):
        if os.path.exists(path):
            return path
    return ""

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    log.info("Starting SWIM pipeline → output: %s", OUTPUT_DIR)
    log.info("Connecting: host=%s  vpn=%s  queue=%s",
             CREDS["host"], CREDS["vpn"], CREDS["queue"])

    if args.no_verify_tls:
        log.warning("TLS verification DISABLED — testing only.")
        tls_props = {
            "solace.messaging.tls.cert-validated":           False,
            "solace.messaging.tls.cert-validate-servername": False,
        }
    else:
        ts = _get_trust_store()
        tls_props = {
            "solace.messaging.tls.cert-validated":           bool(ts),
            "solace.messaging.tls.cert-validate-servername": bool(ts),
            "solace.messaging.tls.trust-store-path":         ts,
        }

    svc = (
        MessagingService.builder()
        .from_properties({
            "solace.messaging.transport.host":
                CREDS["host"],
            "solace.messaging.service.vpn-name":
                CREDS["vpn"],
            "solace.messaging.authentication.scheme.basic.username":
                CREDS["username"],
            "solace.messaging.authentication.scheme.basic.password":
                CREDS["password"],
            **tls_props,
        })
        .build()
    )

    try:
        svc.connect()
        log.info("Connected to SCDS.")
    except Exception as e:
        sys.exit(f"ERROR: {e}")

    queue   = Queue.durable_exclusive_queue(CREDS["queue"])
    handler = PipelineHandler(
        flush_every  = args.flush_every,
        dry_run      = args.dry_run,
        max_records  = args.max_records,
    )
    receiver: PersistentMessageReceiver = (
        svc.create_persistent_message_receiver_builder().build(queue)
    )

    def _shutdown(sig, frame):
        log.info("Shutdown signal — flushing and exiting.")
        handler._done = True

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        receiver.start()
        receiver.receive_async(handler)
        log.info("Pipeline running... (Ctrl-C to stop)")

        deadline = time.time() + args.duration if args.duration else None
        while not handler.done:
            if deadline and time.time() > deadline:
                log.info("Duration limit of %ds reached.", args.duration)
                break
            time.sleep(1.0)

    finally:
        handler.flush_remaining()
        log.info(
            "Pipeline stopped. Total: msgs=%d  records=%d  files written=%d",
            handler.msg_count, handler.total_written, handler._file_index
        )
        try: receiver.terminate()
        except Exception: pass
        try: svc.disconnect()
        except Exception: pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FAA SWIM continuous flight delay feature pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--duration",    type=int, default=0,
                   help="Stop after N seconds (default: 0=run forever)")
    p.add_argument("--max-records", type=int, default=0,
                   help="Stop after N total records (default: 0=unlimited)")
    p.add_argument("--flush-every", type=int, default=500,
                   help="Flush buffer to parquet every N records (default: 500)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print records to stdout, do not write parquet")
    p.add_argument("--no-verify-tls", action="store_true",
                   help="Disable TLS certificate verification (testing only)")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if "YOUR_" in CREDS["host"] or "YOUR_" in CREDS["username"]:
        log.error(
            "Credentials are placeholders. Set environment variables:\n"
            "  export SCDS_HOST=tcps://ems2.swim.faa.gov:55443\n"
            "  export SCDS_USERNAME=jwilso87.gmu.edu\n"
            "  export SCDS_PASSWORD=your_password\n"
            "  export SCDS_QUEUE_FLIGHT=jwilso87.gmu.edu.TFMS.UUID.OUT"
        )
        sys.exit(1)

    run_pipeline(args)

"""
Running:
# Test for 60 seconds printing to screen
python swim_pipeline.py --duration 60 --dry-run --no-verify-tls

# Run for 1 hour writing parquet files
python swim_pipeline.py --duration 3600 --no-verify-tls

# Run continuously flushing every 200 records
python swim_pipeline.py --flush-every 200 --no-verify-tls


Example Data:

<ns2:fltdMessage 
    acid="ABC123"           ← callsign/flight number
    airline="ABC"            ← carrier
    arrArpt="KCLE"           ← arrival airport
    depArpt="KMEM"           ← departure airport
    flightRef="123456789"    ← flight index
    msgType="trackInformation"  ← message type
    sourceTimeStamp="2026-04-18T09:06:50Z">

  <ns2:departureInformation>
    <ns3:timeOfDeparture estimated="false">2026-04-18T09:06:00Z</ns3:timeOfDeparture>
    <ns3:ncsmFlightTimeData>
      <ns3:etd etdType="ACTUAL" timeValue="2026-04-18T09:06:00Z" />
      <ns3:eta etaType="ESTIMATED" timeValue="2026-04-18T10:16:15Z" />
    </ns3:ncsmFlightTimeData>
  </ns2:departureInformation>

  <ns4:gufi>KI123kh</ns4:gufi>       ← globally unique flight ID
  <ns4:igtd>2026-04-18T08:46:00Z</ns4:igtd>  ← scheduled departure
  <ns4:departurePoint><ns4:airport>KSDF</ns4:airport></ns4:departurePoint>
  <ns4:arrivalPoint><ns4:airport>KMDT</ns4:airport></ns4:arrivalPoint>

"""