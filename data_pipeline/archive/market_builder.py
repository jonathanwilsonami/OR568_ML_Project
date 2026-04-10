from __future__ import annotations

from collections import deque
from pathlib import Path

import polars as pl

from utils import ensure_dir


def _build_inbound_hop_map(
    route_time_df: pl.DataFrame,
    core_airports: list[str],
    max_hops: int = 2,
) -> dict[str, int]:
    """
    Build minimum inbound hop distance to the core market.
    Reverse graph logic:
      if A -> B exists, then A is one hop upstream of B.
    """
    reverse_adj: dict[str, set[str]] = {}

    for row in route_time_df.select(["Origin", "Dest"]).unique().iter_rows(named=True):
        origin = row["Origin"]
        dest = row["Dest"]
        reverse_adj.setdefault(dest, set()).add(origin)

    hop_map: dict[str, int] = {}
    q = deque()

    for airport in core_airports:
        hop_map[airport] = 0
        q.append(airport)

    while q:
        node = q.popleft()
        current_hop = hop_map[node]

        if current_hop >= max_hops:
            continue

        for upstream in reverse_adj.get(node, set()):
            new_hop = current_hop + 1
            if upstream not in hop_map or new_hop < hop_map[upstream]:
                hop_map[upstream] = new_hop
                q.append(upstream)

    return hop_map


def build_market_subset(
    flights_canonical_df: pl.DataFrame,
    route_time_df: pl.DataFrame,
    core_airports: list[str],
    max_hops: int,
    out_dir: Path,
    market_name: str,
    year_label: str,
) -> pl.DataFrame:
    ensure_dir(out_dir)

    hop_map = _build_inbound_hop_map(route_time_df, core_airports, max_hops=max_hops)

    hop_df = pl.DataFrame(
        {
            "airport": list(hop_map.keys()),
            "hop_to_core": list(hop_map.values()),
        }
    )

    flights = (
        flights_canonical_df
        .join(
            hop_df.rename({"airport": "Origin", "hop_to_core": "origin_hop_to_core"}),
            on="Origin",
            how="left",
        )
        .join(
            hop_df.rename({"airport": "Dest", "hop_to_core": "dest_hop_to_core"}),
            on="Dest",
            how="left",
        )
    )

    subset = flights.filter(
        (
            (pl.col("origin_hop_to_core") == 0) & (pl.col("dest_hop_to_core") == 0)
        )
        | (
            pl.col("origin_hop_to_core").is_not_null()
            & pl.col("dest_hop_to_core").is_not_null()
            & (pl.col("origin_hop_to_core") == pl.col("dest_hop_to_core") + 1)
        )
    )

    out_path = out_dir / f"{market_name}_h{max_hops}_{year_label}.parquet"
    subset.write_parquet(out_path)
    print(f"Wrote market subset -> {out_path}")
    print(f"Market subset rows: {subset.height:,}")

    return subset


"""
Usage: 
import polars as pl
from pathlib import Path
from market_builder import build_market_subset

flights_canonical = pl.read_parquet("data/features/flights_canonical_2019.parquet")
route_time = pl.read_parquet("data/features/route_time_2019.parquet")

build_market_subset(
    flights_canonical_df=flights_canonical,
    route_time_df=route_time,
    core_airports=["BWI", "JFK", "LGA", "EWR"],
    max_hops=2,
    out_dir=Path("data/markets"),
    market_name="bwi_ny",
    year_label="2019",
)
"""