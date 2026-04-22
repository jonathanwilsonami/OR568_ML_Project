"""
FAA SWIM SCDS Flight Delay Data Sampler
========================================
Connects to the FAA SWIM Cloud Distribution Service (SCDS) via the
Solace native Python API and samples TFMData flight messages to extract
delay-relevant features for ST-GCN model input.

PREREQUISITES
-------------
1. Register at https://scds.faa.gov and request a TFMData subscription.
2. After approval you will receive:
      - JMS Connection URL  (e.g. tcps://host.swim.faa.gov:55443)
      - Message VPN         (e.g. TFMS)
      - Connection Username
      - Connection Password
      - Queue Name          (e.g. username.TFMS.<UUID>.OUT)
3. Install dependencies:
      pip install solace-pubsubplus pandas certifi
4. Download the LADD file from https://adx.faa.gov and place it next
   to this script as IndustryLADD.csv (or set LADD_PATH env var).

ENVIRONMENT VARIABLES
---------------------
    SCDS_HOST_2             tcps://your-host.swim.faa.gov:55443
    SCDS_VPN              TFMS
    SCDS_USERNAME         your_username
    SCDS_PASSWORD         your_password
    SCDS_QUEUE_FLIGHT            your_username.TFMS.UUID.OUT
    LADD_PATH             /path/to/IndustryLADD.csv       (optional)
    SOLACE_TRUST_STORE    /path/to/custom/truststore.pem  (optional)

USAGE
-----
    # Dump first 3 raw messages to see actual XML structure
    python swim_extract.py --debug-raw 3 --no-verify-tls

    # Sample 200 messages and write to CSV
    python swim_extract.py --sample 200 --output swim_sample.csv

    # Run for 60 seconds regardless of message count
    python swim_extract.py --duration 60 --output swim_sample.csv

    # Dry run — print parsed messages to stdout without saving
    python swim_extract.py --sample 10 --dry-run

    # Skip TLS certificate verification (testing only)
    python swim_extract.py --sample 10 --dry-run --no-verify-tls
"""

import argparse
import csv
import json
import logging
import os
import pandas as pd
import signal
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Solace Python API  (pip install solace-pubsubplus)
# ---------------------------------------------------------------------------
try:
    from solace.messaging.messaging_service import MessagingService
    from solace.messaging.resources.queue import Queue
    from solace.messaging.receiver.persistent_message_receiver import PersistentMessageReceiver
    from solace.messaging.receiver.message_receiver import MessageHandler
except ImportError:
    sys.exit(
        "ERROR: Solace Python API not installed.\n"
        "Run:  pip install solace-pubsubplus"
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("swim_extract")

# ---------------------------------------------------------------------------
# SCDS credentials — override via environment variables or CLI flags
# ---------------------------------------------------------------------------
DEFAULT_CREDS = {
    "host":     os.environ.get("SCDS_HOST_2",     "tcps://YOUR_HOST.swim.faa.gov:55443"),
    "vpn":      os.environ.get("SCDS_VPN",      "TFMS"),
    "username": os.environ.get("SCDS_USERNAME", "YOUR_USERNAME"),
    "password": os.environ.get("SCDS_PASSWORD", "YOUR_PASSWORD"),
    "queue":    os.environ.get("SCDS_QUEUE_FLIGHT",    "YOUR_USERNAME.TFMS.UUID.OUT"),
}

# ---------------------------------------------------------------------------
# LADD filter
# ---------------------------------------------------------------------------
# TODO : Need to auto update this file every first Thursday of every month. 
# You have 5 business days to update your copy
_LADD_PATH = os.environ.get("LADD_PATH", "IndustryLADD.csv")
try:
    LADD_SET = set(
        pd.read_csv(_LADD_PATH)["registration"].str.upper().str.strip()
    )
    log.info("Loaded LADD filter: %d blocked registrations from %s",
             len(LADD_SET), _LADD_PATH)
except FileNotFoundError:
    log.warning(
        "LADD file not found at '%s' — no aircraft will be filtered. "
        "Download the IndustryLADD list from https://adx.faa.gov and place "
        "it next to this script, or set the LADD_PATH environment variable.",
        _LADD_PATH)
    LADD_SET = set()
except Exception as e:
    log.warning("Could not load LADD file: %s — continuing without filter", e)
    LADD_SET = set()

# ---------------------------------------------------------------------------
# Message types we care about
# ---------------------------------------------------------------------------
RELEVANT_TYPES = {
    "flightTimes",
    "flightDeparture",
    "flightArrival",
    "flightPlan",
    "flightPlanAmendment",
    "flightCreate",
    "flightModify",
    "flightRoute",
    "flightSchedule",
}


# ---------------------------------------------------------------------------
# Parsed flight record
# ---------------------------------------------------------------------------
@dataclass
class FlightRecord:
    sampled_at:          str             = ""
    acid:                str             = ""
    flight_index:        str             = ""
    origin:              str             = ""
    destination:         str             = ""
    carrier:             str             = ""
    etd_scheduled:       str             = ""
    etd_estimated:       str             = ""
    eta_scheduled:       str             = ""
    eta_estimated:       str             = ""
    departure_delay_min: Optional[float] = None
    arrival_delay_min:   Optional[float] = None
    edct:                str             = ""
    tmi_id:              str             = ""
    aircraft_type:       str             = ""
    flight_status:       str             = ""
    raw_msg_type:        str             = ""
    parse_error:         str             = ""


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_text(element: ET.Element, path: str) -> str:
    node = element.find(path)
    return node.text.strip() if node is not None and node.text else ""


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _delay_minutes(scheduled: str, estimated: str) -> Optional[float]:
    s = _parse_iso(scheduled)
    e = _parse_iso(estimated)
    if s and e:
        return round((e - s).total_seconds() / 60.0, 1)
    return None


def _get_child_tag(root: ET.Element) -> tuple:
    """Return (local_tag, element) of first non-status child of tfmDataService."""
    for child in root:
        local = _strip_ns(child.tag)
        if local != "tfmsStatusOutput":
            return local, child
    for child in root:
        return _strip_ns(child.tag), child
    return "", None


def parse_tfmdata_message(xml_text: str) -> Optional[FlightRecord]:
    rec = FlightRecord(sampled_at=datetime.now(timezone.utc).isoformat())

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        rec.parse_error = f"XML parse error: {e}"
        return rec

    root_tag = _strip_ns(root.tag)

    if root_tag == "tfmDataService":
        msg_type, flight_elem = _get_child_tag(root)
    else:
        msg_type   = root_tag
        flight_elem = root

    rec.raw_msg_type = msg_type

    if msg_type not in RELEVANT_TYPES:
        return None

    if flight_elem is None:
        rec.parse_error = "No flight element found"
        return rec

    try:
        def find(tag_local: str) -> str:
            for elem in flight_elem.iter():
                if _strip_ns(elem.tag) == tag_local:
                    return elem.text.strip() if elem.text else ""
            return ""

        rec.acid          = find("acid") or find("aircraftId") or find("callSign")
        rec.flight_index  = find("flightIndex") or find("gufi") or find("flightRef")
        rec.carrier       = find("carrier") or find("airlineCode") or find("operatingAirline")
        rec.aircraft_type = find("aircraftType") or find("acType")
        rec.flight_status = find("flightStatus") or find("status")
        rec.tmi_id        = find("tmiId") or find("tmiIdentifier")
        rec.edct          = find("edct") or find("expectedDepartureClearanceTime")

        for tag in ("departurePoint", "origin", "dep", "departureAerodrome",
                    "departureAirport", "depArpt"):
            v = find(tag)
            if v:
                rec.origin = v
                break

        for tag in ("arrivalPoint", "destination", "arr", "arrivalAerodrome",
                    "arrivalAirport", "arrArpt"):
            v = find(tag)
            if v:
                rec.destination = v
                break

        for tag in ("etdScheduled", "scheduledDeparture", "stdGate",
                    "scheduledGateDeparture", "stdOut"):
            v = find(tag)
            if v:
                rec.etd_scheduled = v
                break

        for tag in ("etdEstimated", "estimatedDeparture", "actualDeparture",
                    "etdActual", "actualOut", "atdOut"):
            v = find(tag)
            if v:
                rec.etd_estimated = v
                break

        for tag in ("etaScheduled", "scheduledArrival", "staGate",
                    "scheduledGateArrival", "staIn"):
            v = find(tag)
            if v:
                rec.eta_scheduled = v
                break

        for tag in ("etaEstimated", "estimatedArrival", "actualArrival",
                    "etaActual", "actualIn", "ataIn"):
            v = find(tag)
            if v:
                rec.eta_estimated = v
                break

        rec.departure_delay_min = _delay_minutes(rec.etd_scheduled, rec.etd_estimated)
        rec.arrival_delay_min   = _delay_minutes(rec.eta_scheduled,  rec.eta_estimated)

    except Exception as e:
        rec.parse_error = f"Field extraction error: {e}"

    return rec


# ---------------------------------------------------------------------------
# Raw debug handler — dumps raw XML, no parsing
# ---------------------------------------------------------------------------

class RawDebugHandler(MessageHandler):
    def __init__(self, max_msgs: int):
        self.max_msgs = max_msgs
        self.count    = 0
        self._done    = False

    @property
    def done(self) -> bool:
        return self._done

    def on_message(self, message):
        if self._done:
            return
        self.count += 1
        try:
            payload = message.get_payload_as_string()
        except Exception:
            payload = None

        print(f"\n{'='*70}")
        print(f"  RAW MESSAGE {self.count} of {self.max_msgs}")
        print(f"{'='*70}")
        if payload:
            try:
                root = ET.fromstring(payload)
                print(ET.tostring(root, encoding="unicode"))
            except Exception:
                print(payload[:3000])
        else:
            print("  (empty payload)")
        print(f"{'='*70}\n")

        if self.count >= self.max_msgs:
            self._done = True


# ---------------------------------------------------------------------------
# Normal flight message handler
# ---------------------------------------------------------------------------

class FlightMessageHandler(MessageHandler):
    def __init__(self, max_samples: int, dry_run: bool = False):
        self.max_samples = max_samples
        self.dry_run     = dry_run
        self.records:    list[FlightRecord] = []
        self.msg_count   = 0
        self.skip_count  = 0
        self.ladd_count  = 0
        self.error_count = 0
        self._done       = False

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
            payload = None

        if not payload:
            self.skip_count += 1
            return

        rec = parse_tfmdata_message(payload)

        if rec is None:
            self.skip_count += 1
            return

        if rec.parse_error:
            self.error_count += 1
            log.debug("Parse error msg %d: %s", self.msg_count, rec.parse_error)
            return

        acid_upper = (rec.acid or "").upper().strip()
        if acid_upper and acid_upper in LADD_SET:
            self.ladd_count += 1
            self.skip_count += 1
            return

        self.records.append(rec)

        if self.dry_run:
            d = {k: v for k, v in asdict(rec).items() if v not in (None, "")}
            print(json.dumps(d, indent=2))

        count = len(self.records)
        if count % 25 == 0:
            log.info("Collected %d records (msgs: %d  skipped: %d  ladd: %d  errors: %d)",
                     count, self.msg_count, self.skip_count,
                     self.ladd_count, self.error_count)

        if self.max_samples and count >= self.max_samples:
            log.info("Reached target of %d records.", self.max_samples)
            self._done = True


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(records: list[FlightRecord], path: str):
    if not records:
        log.warning("No records to write.")
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    log.info("Wrote %d records to %s", len(records), path)


def print_summary(records: list[FlightRecord], handler: FlightMessageHandler):
    if not records:
        print("\n--- No records collected ---")
        return

    delays_dep = [r.departure_delay_min for r in records if r.departure_delay_min is not None]
    delays_arr = [r.arrival_delay_min   for r in records if r.arrival_delay_min   is not None]
    airports   = (set(r.origin for r in records if r.origin) |
                  set(r.destination for r in records if r.destination))

    print("\n" + "=" * 52)
    print("  SWIM SAMPLE SUMMARY")
    print("=" * 52)
    print(f"  Total messages received  : {handler.msg_count}")
    print(f"  Flight records collected : {len(records)}")
    print(f"  Skipped (non-delay type) : {handler.skip_count}")
    print(f"  Filtered by LADD         : {handler.ladd_count}")
    print(f"  Parse errors             : {handler.error_count}")
    print(f"  Unique airports seen     : {len(airports)}")

    if delays_dep:
        avg = sum(delays_dep) / len(delays_dep)
        print(f"\n  Departure delays ({len(delays_dep)} flights):")
        print(f"    Mean : {avg:.1f} min")
        print(f"    Min  : {min(delays_dep):.1f} min")
        print(f"    Max  : {max(delays_dep):.1f} min")

    if delays_arr:
        avg = sum(delays_arr) / len(delays_arr)
        print(f"\n  Arrival delays ({len(delays_arr)} flights):")
        print(f"    Mean : {avg:.1f} min")
        print(f"    Min  : {min(delays_arr):.1f} min")
        print(f"    Max  : {max(delays_arr):.1f} min")

    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# TLS helper
# ---------------------------------------------------------------------------

def _get_trust_store() -> str:
    override = os.environ.get("SOLACE_TRUST_STORE")
    if override and os.path.exists(override):
        log.info("Using trust store: %s", override)
        return override
    try:
        import certifi
        path = certifi.where()
        log.info("Using certifi trust store: %s", path)
        return path
    except ImportError:
        pass
    for path in ("/etc/ssl/certs/ca-certificates.crt",
                 "/etc/pki/tls/certs/ca-bundle.crt",
                 "/etc/ssl/ca-bundle.pem",
                 "/etc/ssl/cert.pem"):
        if os.path.exists(path):
            log.info("Using system trust store: %s", path)
            return path
    return ""


# ---------------------------------------------------------------------------
# Shared connection builder
# ---------------------------------------------------------------------------

def _build_service(args) -> MessagingService:
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

    return (
        MessagingService.builder()
        .from_properties({
            "solace.messaging.transport.host":                       args.host,
            "solace.messaging.service.vpn-name":                     args.vpn,
            "solace.messaging.authentication.scheme.basic.username": args.username,
            "solace.messaging.authentication.scheme.basic.password": args.password,
            **tls_props,
        })
        .build()
    )


# ---------------------------------------------------------------------------
# Debug mode — dump raw XML
# ---------------------------------------------------------------------------

def run_debug(args):
    log.info("DEBUG MODE — dumping first %d raw messages", args.debug_raw)
    log.info("Connecting: host=%s  vpn=%s", args.host, args.vpn)

    svc = _build_service(args)
    try:
        svc.connect()
        log.info("Connected.")
    except Exception as e:
        sys.exit(f"ERROR: {e}")

    queue    = Queue.durable_exclusive_queue(args.queue)
    handler  = RawDebugHandler(max_msgs=args.debug_raw)
    receiver = svc.create_persistent_message_receiver_builder().build(queue)

    def _shutdown(sig, frame):
        handler._done = True

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        receiver.start()
        receiver.receive_async(handler)
        log.info("Waiting for messages... (Ctrl-C to stop)")
        deadline = time.time() + 120
        while not handler.done and time.time() < deadline:
            time.sleep(0.5)
        if not handler.done:
            log.warning("Timed out. Got %d of %d messages.", handler.count, args.debug_raw)
    finally:
        try: receiver.terminate()
        except Exception: pass
        try: svc.disconnect()
        except Exception: pass


# ---------------------------------------------------------------------------
# Normal sampling mode
# ---------------------------------------------------------------------------

def run_sampler(args):
    log.info("Connecting: host=%s  vpn=%s  queue=%s", args.host, args.vpn, args.queue)

    svc = _build_service(args)
    try:
        svc.connect()
        log.info("Connected to SCDS successfully.")
    except Exception as e:
        sys.exit(f"ERROR: Could not connect.\n{e}")

    queue    = Queue.durable_exclusive_queue(args.queue)
    handler  = FlightMessageHandler(max_samples=args.sample, dry_run=args.dry_run)
    receiver: PersistentMessageReceiver = (
        svc.create_persistent_message_receiver_builder().build(queue)
    )

    def _shutdown(sig, frame):
        log.info("Stopping.")
        handler._done = True

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        receiver.start()
        receiver.receive_async(handler)
        log.info("Receiving messages... (Ctrl-C to stop early)")
        deadline = time.time() + args.duration if args.duration else None
        while not handler.done:
            if deadline and time.time() > deadline:
                log.info("Duration limit of %ds reached.", args.duration)
                break
            time.sleep(0.5)
    finally:
        log.info("Stopping receiver and disconnecting.")
        try: receiver.terminate()
        except Exception: pass
        try: svc.disconnect()
        except Exception: pass

    print_summary(handler.records, handler)

    if not args.dry_run and handler.records:
        write_csv(handler.records, args.output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample FAA SWIM SCDS TFMData messages for flight delay research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",     default=DEFAULT_CREDS["host"])
    p.add_argument("--vpn",      default=DEFAULT_CREDS["vpn"])
    p.add_argument("--username", default=DEFAULT_CREDS["username"])
    p.add_argument("--password", default=DEFAULT_CREDS["password"])
    p.add_argument("--queue",    default=DEFAULT_CREDS["queue"])
    p.add_argument("--sample",   type=int, default=100,
                   help="Stop after N flight records (default: 100, 0=unlimited)")
    p.add_argument("--duration", type=int, default=0,
                   help="Stop after N seconds (default: 0=no limit)")
    p.add_argument("--output",   default="swim_sample.csv")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print records to stdout, do not write CSV")
    p.add_argument("--debug-raw", type=int, default=0, metavar="N",
                   help="Dump first N raw messages as XML and exit — "
                        "use this to inspect the actual schema")
    p.add_argument("--no-verify-tls", action="store_true",
                   help="Disable TLS certificate verification (testing only)")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if "YOUR_" in args.host or "YOUR_" in args.username:
        log.error(
            "Credentials are still placeholders. Set via environment variables:\n"
            "  export SCDS_HOST_2=tcps://your-host.swim.faa.gov:55443\n"
            "  export SCDS_USERNAME=your_username\n"
            "  export SCDS_PASSWORD=your_password\n"
            "  export SCDS_QUEUE_FLIGHT=your_username.TFMS.UUID.OUT"
        )
        sys.exit(1)

    if args.debug_raw > 0:
        run_debug(args)
    else:
        run_sampler(args)