from __future__ import annotations

import random
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_retry_session(
    max_retries: int = 6,
    user_agent: str = "OR568-Pipeline/1.0",
) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=0.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )
    return session


def download_file_with_backoff(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout: int = 180,
    verify_ssl: bool = True,
    max_retries: int = 6,
    backoff_base_seconds: float = 2.0,
) -> None:
    if out_path.exists():
        print(f"Already exists -> {out_path}")
        return

    ensure_dir(out_path.parent)

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading: {url}")
            with session.get(url, stream=True, timeout=timeout, verify=verify_ssl) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            print(f"Saved -> {out_path}")
            return

        except requests.exceptions.RequestException as exc:
            last_exc = exc

            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass

            if attempt == max_retries:
                break

            sleep_s = backoff_base_seconds * (2 ** (attempt - 1))
            sleep_s += random.uniform(0.0, 0.75)

            print(
                f"Download failed (attempt {attempt}/{max_retries}) for {url}. "
                f"Retrying in {sleep_s:.1f}s. Error: {exc}"
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def extract_first_csv(zip_path: Path, extract_dir: Path) -> Path:
    ensure_dir(extract_dir)

    existing_csvs = sorted(extract_dir.rglob("*.csv"))
    if existing_csvs:
        print(f"Using existing extracted CSV -> {existing_csvs[0]}")
        return existing_csvs[0]

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csv_files = sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {extract_dir}")

    return csv_files[0]


def cleanup_path(path: Path) -> None:
    try:
        if not path.exists():
            return

        if path.is_file():
            path.unlink()
            return

        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            except Exception:
                pass

        try:
            path.rmdir()
        except Exception:
            pass
    except Exception:
        pass