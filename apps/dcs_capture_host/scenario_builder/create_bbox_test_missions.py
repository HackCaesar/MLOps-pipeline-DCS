"""Generate bbox-regression DCS missions with varied objects and weather."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Type

from dcs.helicopters import AH_64D, UH_1H, Ka_50, Mi_8MT, Mi_24P
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
    wind_ground_speed: int = 0


@dataclass(frozen=True)
class MissionSpec:
    name: str
    weather: WeatherSpec
    ships: Sequence[ShipSpec]
    aircraft: Sequence[AirSpec]


WEATHER = {
    "clear": WeatherSpec("clear", 12, 0, 300, 200, 80000, wind_ground_speed=1),
    "cloudy": WeatherSpec("cloudy", 10, 5, 900, 700, 65000, wind_ground_speed=4),
    "haze": WeatherSpec("haze", 16, 2, 1200, 500, 22000, wind_ground_speed=2),
    "rain": WeatherSpec("rain", 9, 8, 500, 900, 28000, fog_visibility=12000, fog_thickness=180, wind_ground_speed=7),
    "sunset": WeatherSpec("sunset", 18, 4, 1400, 600, 50000, wind_ground_speed=3),
}


MISSIONS = [
    MissionSpec(
        name="bbox_test_01_clear_mixed",
        weather=WEATHER["clear"],
        ships=(
            ShipSpec("ship_burke_front", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 90),
            ShipSpec("ship_cargo_left", Dry_cargo_ship_1, BASE_X + 260, BASE_Y - 180, 35),
        ),
        aircraft=(
            AirSpec("helo_uh1_near_left", UH_1H, BASE_X - 95, BASE_Y + 60, 145, 80, "USA"),
            AirSpec("plane_a10_high_right", A_10C, BASE_X + 80, BASE_Y - 135, 220, 180, "USA"),
        ),
    ),
    MissionSpec(
        name="bbox_test_02_cloudy_ship_heavy",
        weather=WEATHER["cloudy"],
        ships=(
            ShipSpec("ship_burke_side", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 15),
            ShipSpec("ship_kilo_low", KILO, BASE_X - 180, BASE_Y + 90, 110, "Russia"),
            ShipSpec("ship_cargo_far", Dry_cargo_ship_1, BASE_X + 360, BASE_Y + 180, 240),
        ),
        aircraft=(
            AirSpec("helo_mi8_mid", Mi_8MT, BASE_X + 65, BASE_Y + 155, 170, 90, "Russia"),
            AirSpec("plane_f16_small", F_16C_50, BASE_X - 210, BASE_Y - 120, 260, 220, "USA"),
        ),
    ),
    MissionSpec(
        name="bbox_test_03_haze_air_heavy",
        weather=WEATHER["haze"],
        ships=(
            ShipSpec("ship_burke_oblique", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 310),
        ),
        aircraft=(
            AirSpec("helo_ka50_close", Ka_50, BASE_X - 70, BASE_Y - 80, 125, 95, "Russia"),
            AirSpec("helo_mi24_cross", Mi_24P, BASE_X + 120, BASE_Y + 75, 185, 105, "Russia"),
            AirSpec("plane_su25_low", Su_25T, BASE_X + 40, BASE_Y - 170, 210, 190, "Russia"),
            AirSpec("plane_mig29_far", MiG_29A, BASE_X - 260, BASE_Y + 165, 320, 240, "Russia"),
        ),
    ),
    MissionSpec(
        name="bbox_test_04_rain_close_overlap",
        weather=WEATHER["rain"],
        ships=(
            ShipSpec("ship_burke_close", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 180),
            ShipSpec("ship_cargo_partial", Dry_cargo_ship_1, BASE_X - 120, BASE_Y - 85, 210),
        ),
        aircraft=(
            AirSpec("helo_apache_above", AH_64D, BASE_X + 35, BASE_Y + 35, 120, 90, "USA"),
            AirSpec("plane_a10_low", A_10C, BASE_X - 45, BASE_Y + 125, 170, 170, "USA"),
        ),
    ),
    MissionSpec(
        name="bbox_test_05_sunset_extreme_angles",
        weather=WEATHER["sunset"],
        ships=(
            ShipSpec("ship_burke_stern", USS_Arleigh_Burke_IIa, BASE_X, BASE_Y, 270),
            ShipSpec("ship_kilo_side", KILO, BASE_X + 210, BASE_Y + 120, 25, "Russia"),
        ),
        aircraft=(
            AirSpec("helo_uh1_right", UH_1H, BASE_X + 140, BASE_Y - 40, 155, 80, "USA"),
            AirSpec("helo_ka50_left", Ka_50, BASE_X - 160, BASE_Y + 35, 190, 95, "Russia"),
            AirSpec("plane_f16_high", F_16C_50, BASE_X + 45, BASE_Y + 210, 350, 230, "USA"),
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


def build_test_mission(spec: MissionSpec, output_path: Path) -> Path:
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
    parser = argparse.ArgumentParser(description="Generate bbox regression test missions")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "missions" / "bbox_tests"),
        help="Directory for generated .miz files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    generated = []
    for spec in MISSIONS:
        generated.append(str(build_test_mission(spec, output_dir / f"{spec.name}.miz")))
    print("\n".join(generated))


if __name__ == "__main__":
    main()
