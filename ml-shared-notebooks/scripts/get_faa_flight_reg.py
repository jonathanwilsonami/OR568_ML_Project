#!/usr/bin/env python3
"""
FAA Aircraft Registry loader for joining with OpenSky (icao24).

- Downloads FAA ReleasableAircraft.zip (cached, streaming, retries).
- Parses ACFTREF, ENGINE, MASTER using faa-aircraft-registry package internals.
- Builds a join-ready dimension keyed by icao24 (Mode S / transponder_code_hex).
- Writes output to data/ as parquet + csv (optional).

Run:
  python scripts/get_faa_registry.py

Outputs:
  data/faa_aircraft_dim.parquet
  data/faa_aircraft_dim.csv
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from typing import Optional, Dict, Any, Iterable

import requests

# You already have this installed
from faa_aircraft_registry.aircraft import read as read_aircraft
from faa_aircraft_registry.engines import read as read_engines
from faa_aircraft_registry.master import read as read_master

# Optional but recommended for output files
try:
    import pandas as pd
except ImportError:
    pd = None


FAA_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"


# ----------------------------
# Paths / caching
# ----------------------------

def base_dir() -> str:
    """Works in scripts and notebooks."""
    try:
        return os.path.dirname(__file__)
    except NameError:
        return os.getcwd()


OUT_DIR = os.path.abspath(os.path.join(base_dir(), "..", "data"))
ZIP_PATH = os.path.join(OUT_DIR, "ReleasableAircraft.zip")

# How often to re-download (hours)
MAX_AGE_HOURS = 48


# ----------------------------
# Download helper
# ----------------------------

def is_fresh_file(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 5_000_000:  # sanity check (zip should be much larger than this)
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_hours * 3600


def download_with_retries(url: str, out_path: str, attempts: int = 6) -> None:
    """
    Robust streaming download with retries/backoff.
    - Writes to out_path.part then atomic rename.
    - Uses long read timeout because FAA can be slow.
    """
    timeout = (20, 900)  # (connect timeout seconds, read timeout seconds)
    backoff = 5

    headers = {"User-Agent": "OR568-ML-Project/1.0 (FAA Registry downloader)"}

    for i in range(1, attempts + 1):
        tmp_path = out_path + ".part"
        try:
            print(f"Downloading FAA registry (attempt {i}/{attempts})...")
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()

                total = int(r.headers.get("Content-Length") or 0)
                downloaded = 0

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):  # 2MB chunks
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
                        else:
                            print(f"\r  {downloaded/1e6:.1f}MB downloaded", end="")

                os.replace(tmp_path, out_path)
                print("\nDownload complete.")
                return

        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            print(f"\nDownload failed: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if i == attempts:
                raise

            print(f"Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2


def validate_zip(path: str) -> None:
    """Raise if ZIP is corrupt."""
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member detected: {bad}")


# ----------------------------
# Parsing helpers
# ----------------------------

def open_member_case_insensitive(z: zipfile.ZipFile, wanted: str):
    """
    Open a member by filename, case-insensitive, allowing for folder prefixes.
    Example: wanted='MASTER.txt' will match 'MASTER.TXT' or 'ReleasableAircraft/MASTER.txt'.
    """
    wanted_lower = wanted.lower()
    for name in z.namelist():
        if name.lower().endswith(wanted_lower):
            return z.open(name, "r")
    # Give a helpful preview
    preview = "\n".join(z.namelist()[:40])
    raise KeyError(f"Missing '{wanted}'. ZIP contains (first 40):\n{preview}")


def read_faa_zip(z: zipfile.ZipFile) -> Dict[str, Any]:
    """
    Parses the FAA ZIP using the faa-aircraft-registry internal readers.
    Returns dict with keys: aircraft, engines, master (dicts).
    """
    # ACFTREF.txt
    with open_member_case_insensitive(z, "ACFTREF.txt") as f:
        aircraft = read_aircraft(io.TextIOWrapper(f, "utf-8-sig"))

    # ENGINE.txt
    with open_member_case_insensitive(z, "ENGINE.txt") as f:
        engines = read_engines(io.TextIOWrapper(f, "utf-8-sig"))

    # MASTER.txt
    with open_member_case_insensitive(z, "MASTER.txt") as f:
        master = read_master(io.TextIOWrapper(f, "utf-8-sig"), aircraft, engines)

    return {"aircraft": aircraft, "engines": engines, "master": master}


# ----------------------------
# Join-key normalization + dimension building
# ----------------------------

def norm_icao24(x: Optional[str]) -> Optional[str]:
    """
    Normalize FAA transponder_code_hex / Mode S hex into OpenSky-style icao24:
    - lowercase
    - strip whitespace
    - drop '0x' prefix if present
    - zero-pad to 6 chars
    """
    if not x:
        return None
    x = str(x).strip().lower()
    if not x:
        return None
    if x.startswith("0x"):
        x = x[2:]
    # Keep only hex-ish characters (optional strictness)
    x = "".join(ch for ch in x if ch in "0123456789abcdef")
    if not x:
        return None
    return x.zfill(6)


def build_aircraft_dim(master: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Build a compact, join-ready dimension keyed by icao24.

    master: dict keyed by unique_regulatory_id, values = record dict.
    Returns iterable of rows.
    """
    for rec in master.values():
        icao24 = norm_icao24(rec.get("transponder_code_hex"))
        if not icao24:
            continue

        aircraft = rec.get("aircraft") or {}
        engine = rec.get("engine") or {}
        registrant = rec.get("registrant") or {}

        yield {
            # Join key to OpenSky
            "icao24": icao24,

            # FAA identity / registration
            "n_number": rec.get("registration_number"),
            "unique_regulatory_id": rec.get("unique_regulatory_id"),
            "status": rec.get("status"),
            "last_action_date": rec.get("last_action_date"),
            "certificate_issue_date": rec.get("certificate_issue_date"),
            "expiration_date": rec.get("expiration_date"),
            "airworthiness_date": rec.get("airworthiness_date"),

            # Aircraft characteristics (from ACFTREF via aircraft_manufacturer_code)
            "aircraft_code": (aircraft.get("code") if isinstance(aircraft, dict) else None),
            "aircraft_manufacturer": (aircraft.get("manufacturer") if isinstance(aircraft, dict) else None),
            "aircraft_model": (aircraft.get("model") if isinstance(aircraft, dict) else None),
            "aircraft_type": rec.get("aircraft_type"),
            "aircraft_category": (aircraft.get("category") if isinstance(aircraft, dict) else None),
            "aircraft_certification": (aircraft.get("certification") if isinstance(aircraft, dict) else None),
            "weight_category": (aircraft.get("weight_category") if isinstance(aircraft, dict) else None),
            "num_engines": (aircraft.get("number_of_engines") if isinstance(aircraft, dict) else None),
            "num_seats": (aircraft.get("number_of_seats") if isinstance(aircraft, dict) else None),
            "cruising_speed_mph": (aircraft.get("cruising_speed_mph") if isinstance(aircraft, dict) else None),

            # Engine characteristics (from ENGINE via engine_manufacturer_code)
            "engine_code": (engine.get("code") if isinstance(engine, dict) else None),
            "engine_type": rec.get("engine_type"),
            "engine_manufacturer": (engine.get("manufacturer") if isinstance(engine, dict) else None),
            "engine_model": (engine.get("model") if isinstance(engine, dict) else None),
            "engine_power_hp": (engine.get("power_hp") if isinstance(engine, dict) else None),
            "engine_thrust_lbf": (engine.get("thrust_lbf") if isinstance(engine, dict) else None),

            # Manufacturing year
            "manufacturing_year": rec.get("manufacturing_year"),

            # Optional owner-ish fields (be mindful of privacy / use-case)
            "registrant_type": registrant.get("type") if isinstance(registrant, dict) else None,
            "registrant_name": registrant.get("name") if isinstance(registrant, dict) else None,
            "registrant_city": registrant.get("city") if isinstance(registrant, dict) else None,
            "registrant_state": registrant.get("state") if isinstance(registrant, dict) else None,
            "registrant_country": registrant.get("country") if isinstance(registrant, dict) else None,
        }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not is_fresh_file(ZIP_PATH, MAX_AGE_HOURS):
        download_with_retries(FAA_URL, ZIP_PATH)
    else:
        print(f"Using cached FAA registry ZIP: {ZIP_PATH}")

    validate_zip(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as z:
        # Helpful if you ever need to debug filename mismatches:
        # print("ZIP members (first 30):", z.namelist()[:30])

        data = read_faa_zip(z)

    print(f"Parsed: aircraft={len(data['aircraft'])}, engines={len(data['engines'])}, master={len(data['master'])}")

    # Build joinable dimension (icao24)
    dim_rows = list(build_aircraft_dim(data["master"]))
    print(f"Built dim rows with icao24: {len(dim_rows)}")

    if pd is None:
        print("pandas not installed; skipping parquet/csv output.")
        return

    df = pd.DataFrame(dim_rows)

    # Drop duplicates on icao24, keep latest by last_action_date if present
    if "last_action_date" in df.columns:
        df = df.sort_values(["icao24", "last_action_date"], na_position="last")
    df = df.drop_duplicates(subset=["icao24"], keep="last")

    out_parquet = os.path.join(OUT_DIR, "faa_aircraft_dim.parquet")
    out_csv = os.path.join(OUT_DIR, "faa_aircraft_dim.csv")

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"Wrote:\n  {out_parquet}\n  {out_csv}")
    print("Sample:")
    print(df[["icao24", "n_number", "aircraft_manufacturer", "aircraft_model", "engine_type", "manufacturing_year"]].head(10))


if __name__ == "__main__":
    main()
