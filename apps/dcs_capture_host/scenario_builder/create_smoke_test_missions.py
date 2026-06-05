"""Generate small production-like smoke-test missions.

These missions intentionally vary class composition, weather, time of day,
object counts, and headings while staying moderate enough for a bbox smoke test.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Type

from dcs.helicopters import UH_1H, Ka_50, Mi_8MT, Mi_24P
from dcs.mapping import Point
from dcs.mission import Mission
from dcs.planes import A_10C, F_16C_50, MiG_29A, Su_25T
from dcs.ships import KILO, Dry_cargo_ship_1, USS_Arleigh_Burke_IIa
from dcs.terrain import Caucasus
from dcs.unittype import FlyingType, ShipType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_X = -150000.0
BASE_Y = -300000.0


@dataclass(frozen=True)
class ShipSpec:
    name: str
    unit_type: Type[ShipType]
    x: float
    y: float
    heading: float
    country: str = "USA"


@dataclass(frozen=True)
class AirSpec:
    name: str
    unit_type: Type[FlyingType]
    x: float
    y: float
    altitude: int
    speed: int
    country: str


@dataclass(frozen=True)
class WeatherSpec:
    name: str
    start_hour: int
    clouds_density: int
    clouds_base: int
    clouds_thickness: int
    visibility_distance: int
    fog_visibility: int = 25
    fog_thickness: int = 0
    wind_ground_speed: int = 1


@dataclass(frozen=True)
class MissionSpec:
    name: str
    weather: WeatherSpec
    ships: Sequence[ShipSpec]
    aircraft: Sequence[AirSpec]


WEATHER = {
    "morning_clear": WeatherSpec("morning_clear", 8, 0, 400, 200, 80000, wind_ground_speed=1),
    "noon_light_clouds": WeatherSpec("noon_light_clouds", 12, 3, 1000, 500, 70000, wind_ground_speed=2),
    "overcast": WeatherSpec("overcast", 14, 7, 900, 700, 60000, wind_ground_speed=4),
    "haze": WeatherSpec("haze", 16, 2, 1200, 500, 26000, wind_ground_speed=2),
    "low_sun": WeatherSpec("low_sun", 18, 4, 1400, 600, 50000, wind_ground_speed=3),
}


MISSIONS = [
    MissionSpec(
        name="smoke_01_morning_ship_single",
        weather=WEATHER["morning_clear"],
        ships=(ShipSpec("smoke_ship_burke_front_single", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 85),),
        aircraft=(),
    ),
    MissionSpec(
        name="smoke_02_noon_ship_pair",
        weather=WEATHER["noon_light_clouds"],
        ships=(
            ShipSpec("smoke_ship_burke_side", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 15),
            ShipSpec("smoke_ship_cargo_oblique", Dry_cargo_ship_1, BASE_X + 320, BASE_Y - 180, 225),
            ShipSpec("smoke_ship_kilo_far", KILO, BASE_X - 230, BASE_Y + 150, 115, "Russia"),
        ),
        aircraft=(),
    ),
    MissionSpec(
        name="smoke_03_overcast_mixed_medium",
        weather=WEATHER["overcast"],
        ships=(
            ShipSpec("smoke_ship_cargo_left", Dry_cargo_ship_1, BASE_X - 120, BASE_Y + 110, 35),
            ShipSpec("smoke_ship_burke_rear", USS_Arleigh_Burke_IIa, BASE_X + 190, BASE_Y - 120, 270),
        ),
        aircraft=(
            AirSpec("smoke_helo_uh1_mid", UH_1H, BASE_X - 70, BASE_Y - 80, 150, 80, "USA"),
            AirSpec("smoke_plane_a10_mid", A_10C, BASE_X + 80, BASE_Y + 160, 220, 180, "USA"),
        ),
    ),
    MissionSpec(
        name="smoke_04_haze_air_only",
        weather=WEATHER["haze"],
        ships=(),
        aircraft=(
            AirSpec("smoke_helo_ka50_left", Ka_50, BASE_X - 90, BASE_Y - 80, 130, 95, "Russia"),
            AirSpec("smoke_helo_mi24_right", Mi_24P, BASE_X + 140, BASE_Y + 75, 190, 105, "Russia"),
            AirSpec("smoke_plane_su25_low", Su_25T, BASE_X + 50, BASE_Y - 180, 210, 190, "Russia"),
            AirSpec("smoke_plane_mig29_far", MiG_29A, BASE_X - 290, BASE_Y + 185, 320, 240, "Russia"),
        ),
    ),
    MissionSpec(
        name="smoke_05_noon_air_close_pairs",
        weather=WEATHER["noon_light_clouds"],
        ships=(),
        aircraft=(
            AirSpec("smoke_helo_mi8_near", Mi_8MT, BASE_X - 55, BASE_Y + 40, 155, 90, "Russia"),
            AirSpec("smoke_helo_ka50_pair", Ka_50, BASE_X + 15, BASE_Y + 92, 190, 95, "Russia"),
            AirSpec("smoke_plane_f16_pair", F_16C_50, BASE_X + 105, BASE_Y - 40, 275, 230, "USA"),
            AirSpec("smoke_plane_a10_pair", A_10C, BASE_X - 150, BASE_Y - 135, 230, 180, "USA"),
        ),
    ),
    MissionSpec(
        name="smoke_06_low_sun_mixed_oblique",
        weather=WEATHER["low_sun"],
        ships=(
            ShipSpec("smoke_ship_burke_stern", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 270),
            ShipSpec("smoke_ship_kilo_side", KILO, BASE_X + 210, BASE_Y + 120, 25, "Russia"),
        ),
        aircraft=(
            AirSpec("smoke_helo_uh1_right", UH_1H, BASE_X + 140, BASE_Y - 40, 155, 80, "USA"),
            AirSpec("smoke_helo_ka50_left", Ka_50, BASE_X - 160, BASE_Y + 35, 190, 95, "Russia"),
            AirSpec("smoke_plane_f16_high", F_16C_50, BASE_X + 45, BASE_Y + 210, 350, 230, "USA"),
        ),
    ),
]


def apply_weather(mission: Mission, weather: WeatherSpec) -> None:
    mission.start_time = dt.datetime(2026, 4, 22, weather.start_hour, 0, 0)
    mission.weather.name = weather.name
    mission.weather.clouds_density = weather.clouds_density
    mission.weather.clouds_base = weather.clouds_base
    mission.weather.clouds_thickness = weather.clouds_thickness
    mission.weather.visibility_distance = weather.visibility_distance
    mission.weather.fog_visibility = weather.fog_visibility
    mission.weather.fog_thickness = weather.fog_thickness
    mission.weather.enable_fog = weather.fog_thickness > 0
    mission.weather.wind_at_ground.speed = weather.wind_ground_speed


def build_smoke_mission(spec: MissionSpec, output_path: Path) -> Path:
    mission = Mission(Caucasus())
    apply_weather(mission, spec.weather)
    countries = {"USA": mission.country("USA"), "Russia": mission.country("Russia")}

    for ship in spec.ships:
        mission.ship_group(
            countries[ship.country],
            ship.name,
            ship.unit_type,
            Point(ship.x, ship.y, mission.terrain),
            heading=ship.heading,
            group_size=1,
        )

    for aircraft in spec.aircraft:
        mission.flight_group_inflight(
            countries[aircraft.country],
            aircraft.name,
            aircraft.unit_type,
            Point(aircraft.x, aircraft.y, mission.terrain),
            altitude=aircraft.altitude,
            speed=aircraft.speed,
            group_size=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mission.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate production-like smoke-test DCS missions")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "missions" / "smoke_tests"),
        help="Directory for generated .miz files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    generated = []
    for spec in MISSIONS:
        generated.append(str(build_smoke_mission(spec, output_dir / f"{spec.name}.miz")))
    print("\n".join(generated))


if __name__ == "__main__":
    main()
