"""Generate a minimal sea-only DCS mission for the MVP pipeline.

The mission intentionally contains only a few targets over the Black Sea:
- one ship,
- one helicopter in flight,
- one airplane in flight.

This keeps the first end-to-end capture reproducible and easy to debug.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from dcs.helicopters import UH_1H
from dcs.mapping import Point
from dcs.mission import Mission
from dcs.planes import A_10C
from dcs.ships import USS_Arleigh_Burke_IIa
from dcs.terrain import Caucasus


def build_mvp_mission(output_path: Path, mode: str = "mixed") -> Path:
    mission = Mission(Caucasus())
    mission.start_time = dt.datetime(2026, 4, 22, 12, 0, 0)

    usa = mission.country("USA")

    # Coordinates are chosen over the Black Sea near the central-water area used
    # by the earlier prototype. DCS map coordinates here follow pydcs Point(x, y).
    ship_point = Point(-150000.0, -300000.0, mission.terrain)
    if mode == "mixed_close":
        helo_point = Point(-150095.0, -299940.0, mission.terrain)
        plane_point = Point(-150030.0, -300085.0, mission.terrain)
        helo_altitude = 150
        plane_altitude = 180
    else:
        helo_point = Point(-149100.0, -299200.0, mission.terrain)
        plane_point = Point(-148200.0, -298400.0, mission.terrain)
        helo_altitude = 200
        plane_altitude = 280

    mission.ship_group(
        usa,
        "mvp_ship_group",
        USS_Arleigh_Burke_IIa,
        ship_point,
        heading=90,
        group_size=1,
    )

    if mode in {"mixed", "mixed_close"}:
        mission.flight_group_inflight(
            usa,
            "mvp_helicopter_group",
            UH_1H,
            helo_point,
            altitude=helo_altitude,
            speed=80,
            group_size=1,
        )

        mission.flight_group_inflight(
            usa,
            "mvp_airplane_group",
            A_10C,
            plane_point,
            altitude=plane_altitude,
            speed=180,
            group_size=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mission.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal DCS MVP mission")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "missions" / "mvp_sea_scene.miz"),
        help="Output .miz path",
    )
    parser.add_argument(
        "--mode",
        choices=["mixed", "mixed_close", "ship_only"],
        default="mixed",
        help="Mission composition mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    result = build_mvp_mission(output_path, mode=args.mode)
    print(result)


if __name__ == "__main__":
    main()
