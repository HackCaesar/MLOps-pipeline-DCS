"""Generate 10 static neutral Caucasus missions for the 100-frame dataset run.

Strategy (after two audits):

Run 1 (the first AI) placed ships at the open-sea zone (-150000, -300000)
and aircraft at Batumi/Kobuleti airfields hundreds of kilometres away. The
camera planner then averaged the centres of all candidate objects and ended
up looking at empty water in between, giving 40 / 100 invalid frames.

Run 2 (my first rewrite) fixed the orbit math + split into single-class
missions: ships in the open sea, aircraft at airfield parking. Mission 3
(helicopters at Batumi) was rated "очень плохая" because the airfield is
inland and the ship-mounted camera scene becomes irrelevant.

Run 3 (this file) addresses the user's explicit request: the camera mimics a
**ship's onboard camera**, so every observed object must live over the sea.
Aircraft therefore spawn in-air over the same sea cluster as the ships:

  * helicopters: hover (`speed=0`) at 60-120 m AGL, uncontrolled;
  * planes: slow level cruise at 200-300 m AGL with a single waypoint that
    keeps them inside the capture zone (uncontrolled, no aggressive AI).

The DCS pause-before-screenshot hook freezes whichever frame the plane is
mid-pass through, so a slowly cruising plane still gives clean static-looking
shots from the ship's vantage.

All units stay neutral (United Nations Peacekeepers).

Run from any cwd:

    python scenario_builder/create_caucasus_static_neutral_missions.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_pydcs_livery_scan_guard() -> None:
    """Keep pydcs from scanning a real DCS install while generating missions."""
    guard_root = PROJECT_ROOT / ".pydcs_livery_guard"
    home_dir = guard_root / "home"
    dcs_install_dir = guard_root / "DCSWorld"
    saved_games_dir = guard_root / "Saved Games" / "DCS"
    prefs_dir = home_dir / "AppData" / "Local" / "DCSLiberation"
    prefs_path = prefs_dir / "liberation_preferences.json"

    dcs_install_dir.mkdir(parents=True, exist_ok=True)
    saved_games_dir.mkdir(parents=True, exist_ok=True)
    prefs_dir.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(
        json.dumps(
            {
                "dcs_install_dir": str(dcs_install_dir),
                "saved_game_dir": str(saved_games_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.environ["USERPROFILE"] = str(home_dir)
    os.environ["HOME"] = str(home_dir)


configure_pydcs_livery_scan_guard()

import dcs.task as task
from dcs.coalition import Coalition
from dcs.countries import UnitedNationsPeacekeepers
from dcs.helicopters import AH_64D, UH_1H, Ka_50, Mi_8MT, Mi_24P
from dcs.mapping import Point
from dcs.mission import Mission
from dcs.planes import A_10C, MiG_29A, Su_25T
from dcs.ships import KILO, Dry_cargo_ship_1, Dry_cargo_ship_2, Ship_Tilde_Supply
from dcs.terrain import Caucasus
from dcs.unittype import FlyingType, ShipType

# Open-sea anchor (same region used by the original mvp_sea_scene). All units
# in a single mission stay within ~400 m of this point so the camera planner's
# primary target is always inside the frame.
SEA_BASE_X = -150000.0
SEA_BASE_Y = -300000.0


# ----------------------------------------------------------------------- #
# Mission specs
# ----------------------------------------------------------------------- #


@dataclass(frozen=True)
class ShipSpec:
    name: str
    unit_type: Type[ShipType]
    dx: float
    dy: float
    heading: float


@dataclass(frozen=True)
class HelicopterSpec:
    """Helicopter hovering over the sea cluster.

    ``dx``/``dy`` are metres east/north of the sea anchor; ``altitude`` is
    in metres above sea level (the user wanted 150-300 m for airplanes, but
    helicopters look cleaner at 60-120 m where they would normally hover for
    a ship-deck overflight).
    """
    name: str
    unit_type: Type[FlyingType]
    dx: float
    dy: float
    altitude: float = 80.0
    heading: float = 90.0


@dataclass(frozen=True)
class PlaneSpec:
    """Plane in slow level cruise over the sea cluster.

    Same offset/altitude semantics as ``HelicopterSpec``. Speed must stay
    above stall, so we cruise at ``cruise_speed_mps`` (default 90 m/s ≈
    325 km/h) and add one waypoint ``waypoint_dx``/``waypoint_dy`` metres
    along the heading to keep the plane near the capture zone.
    """
    name: str
    unit_type: Type[FlyingType]
    dx: float
    dy: float
    altitude: float = 240.0
    heading: float = 90.0
    cruise_speed_mps: float = 90.0


@dataclass(frozen=True)
class WeatherSpec:
    season: str
    date: dt.datetime
    weather: str
    visibility: str
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
    ships:        Sequence[ShipSpec]        = ()
    helicopters:  Sequence[HelicopterSpec]  = ()
    planes:       Sequence[PlaneSpec]       = ()
    frames: int = 10


# Mission 1 used 08:00 winter morning, which is well before sunrise on the
# Caucasus map in January and produced too-dark frames. Pushing it to 10:00
# keeps the "winter morning" theme but gives the renderer enough light.
WEATHER = {
    "winter_morning_clear": WeatherSpec(
        "winter", dt.datetime(2026, 1, 15, 10, 0, 0),
        "clear", "good",
        clouds_density=0, clouds_base=700, clouds_thickness=200,
        visibility_distance=80000, wind_ground_speed=1,
    ),
    "winter_day_cloudy": WeatherSpec(
        "winter", dt.datetime(2026, 1, 16, 12, 0, 0),
        "cloudy cold light", "good",
        clouds_density=6, clouds_base=900, clouds_thickness=700,
        visibility_distance=65000, wind_ground_speed=4,
    ),
    "spring_morning_haze": WeatherSpec(
        "spring", dt.datetime(2026, 4, 12, 9, 0, 0),
        "light haze", "medium",
        clouds_density=2, clouds_base=1100, clouds_thickness=450,
        visibility_distance=32000, fog_visibility=18000, fog_thickness=90,
        wind_ground_speed=2,
    ),
    "spring_evening_broken": WeatherSpec(
        "spring", dt.datetime(2026, 4, 13, 17, 30, 0),
        "broken clouds", "good",
        clouds_density=4, clouds_base=1300, clouds_thickness=600,
        visibility_distance=55000, wind_ground_speed=3,
    ),
    "summer_day_clear": WeatherSpec(
        "summer", dt.datetime(2026, 7, 18, 12, 0, 0),
        "clear", "excellent",
        clouds_density=0, clouds_base=1000, clouds_thickness=200,
        visibility_distance=90000, wind_ground_speed=1,
    ),
    "summer_evening_warm": WeatherSpec(
        "summer", dt.datetime(2026, 7, 19, 18, 0, 0),
        "warm low sun", "good",
        clouds_density=3, clouds_base=1500, clouds_thickness=550,
        visibility_distance=70000, wind_ground_speed=2,
    ),
    "autumn_morning_overcast": WeatherSpec(
        "autumn", dt.datetime(2026, 10, 8, 9, 0, 0),
        "overcast", "medium",
        clouds_density=8, clouds_base=800, clouds_thickness=900,
        visibility_distance=45000, fog_visibility=16000, fog_thickness=140,
        wind_ground_speed=5,
    ),
    "autumn_day_cloudy": WeatherSpec(
        "autumn", dt.datetime(2026, 10, 9, 12, 0, 0),
        "cloudy", "medium",
        clouds_density=6, clouds_base=1000, clouds_thickness=700,
        visibility_distance=52000, wind_ground_speed=4,
    ),
    "spring_day_mist": WeatherSpec(
        "spring", dt.datetime(2026, 4, 14, 13, 0, 0),
        "light mist", "reduced but clear targets",
        clouds_density=3, clouds_base=900, clouds_thickness=500,
        visibility_distance=30000, fog_visibility=14000, fog_thickness=120,
        wind_ground_speed=2,
    ),
    "autumn_evening_complex": WeatherSpec(
        "autumn", dt.datetime(2026, 10, 10, 16, 30, 0),
        "evening broken haze", "medium",
        clouds_density=5, clouds_base=1200, clouds_thickness=650,
        visibility_distance=42000, fog_visibility=17000, fog_thickness=90,
        wind_ground_speed=3,
    ),
}


# Ship-only missions: 2-3 ships clustered within ~400 m. The camera planner
# picks the highest-priority ship as primary and orbits around it.
SHIP_MISSIONS: Sequence[MissionSpec] = (
    MissionSpec(
        name="caucasus_static_neutral_01_winter_morning_clear_ships",
        weather=WEATHER["winter_morning_clear"],
        ships=(
            ShipSpec("ship_01_cargo_lead", Dry_cargo_ship_1, dx=0.0,   dy=0.0,   heading=85),
            ShipSpec("ship_01_kilo_side",  KILO,             dx=140.0, dy=-90.0, heading=35),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_02_winter_day_cloudy_ships",
        weather=WEATHER["winter_day_cloudy"],
        ships=(
            ShipSpec("ship_02_cargo_a", Dry_cargo_ship_2,  dx=-110.0, dy=80.0,   heading=145),
            ShipSpec("ship_02_cargo_b", Dry_cargo_ship_1,  dx=170.0,  dy=130.0,  heading=255),
            ShipSpec("ship_02_supply",  Ship_Tilde_Supply, dx=-20.0,  dy=-160.0, heading=20),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_05_summer_day_clear_ships",
        weather=WEATHER["summer_day_clear"],
        ships=(
            ShipSpec("ship_05_kilo",   KILO,              dx=0.0,    dy=0.0,    heading=270),
            ShipSpec("ship_05_cargo",  Dry_cargo_ship_1,  dx=-220.0, dy=100.0,  heading=110),
            ShipSpec("ship_05_supply", Ship_Tilde_Supply, dx=190.0,  dy=-140.0, heading=15),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_08_autumn_day_cloudy_ships",
        weather=WEATHER["autumn_day_cloudy"],
        ships=(
            ShipSpec("ship_08_cargo_lead",  Dry_cargo_ship_2, dx=-120.0, dy=80.0,   heading=10),
            ShipSpec("ship_08_cargo_trail", Dry_cargo_ship_1, dx=80.0,   dy=-70.0,  heading=190),
            ShipSpec("ship_08_kilo",        KILO,             dx=210.0,  dy=110.0,  heading=95),
        ),
    ),
)


# Helicopter missions: one zero-speed ship-camera platform + helicopters
# hovering above and beside the ship. Camera planner picks the ship as
# primary; helicopters pass through the orbit FOV at known offsets.
HELICOPTER_MISSIONS: Sequence[MissionSpec] = (
    MissionSpec(
        name="caucasus_static_neutral_03_spring_morning_haze_helos",
        weather=WEATHER["spring_morning_haze"],
        ships=(
            ShipSpec("ship_03_cargo_anchor", Dry_cargo_ship_1, dx=0.0, dy=0.0, heading=90),
        ),
        helicopters=(
            HelicopterSpec("helo_03_uh1",  UH_1H,  dx=120.0, dy=80.0,   altitude=90.0,  heading=180),
            HelicopterSpec("helo_03_mi24", Mi_24P, dx=-90.0, dy=150.0,  altitude=110.0, heading=270),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_06_summer_evening_warm_helos",
        weather=WEATHER["summer_evening_warm"],
        ships=(
            ShipSpec("ship_06_supply_anchor", Ship_Tilde_Supply, dx=0.0, dy=0.0, heading=120),
        ),
        helicopters=(
            HelicopterSpec("helo_06_mi8",  Mi_8MT, dx=80.0,   dy=-40.0, altitude=70.0,  heading=200),
            HelicopterSpec("helo_06_ka50", Ka_50,  dx=-110.0, dy=60.0,  altitude=95.0,  heading=320),
            HelicopterSpec("helo_06_uh1",  UH_1H,  dx=60.0,   dy=120.0, altitude=80.0,  heading=160),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_09_spring_day_mist_helos",
        weather=WEATHER["spring_day_mist"],
        ships=(
            ShipSpec("ship_09_kilo_anchor", KILO, dx=0.0, dy=0.0, heading=20),
        ),
        helicopters=(
            HelicopterSpec("helo_09_apache", AH_64D, dx=140.0,  dy=-100.0, altitude=100.0, heading=240),
            HelicopterSpec("helo_09_ka50",   Ka_50,  dx=-160.0, dy=80.0,   altitude=75.0,  heading=70),
        ),
    ),
)


# Plane missions: ship anchor + slowly cruising planes 350-500 m AGL.
#
# Altitude was 200-300 m in run 4 but jets in a Race-Track loiter naturally
# lose 100-150 m of altitude during the banked legs; that puts them in the
# water before the AI recovers. 350-500 m keeps the same "low ship-camera
# scene" character (still well below cloud base) while giving the AI enough
# vertical room to hold the orbit. Cruise speed stays slow (85-110 m/s) so
# the plane sweeps through the capture zone instead of flashing past.
PLANE_MISSIONS: Sequence[MissionSpec] = (
    MissionSpec(
        name="caucasus_static_neutral_04_spring_evening_broken_planes",
        weather=WEATHER["spring_evening_broken"],
        ships=(
            ShipSpec("ship_04_cargo_anchor", Dry_cargo_ship_2, dx=0.0, dy=0.0, heading=70),
        ),
        planes=(
            PlaneSpec("plane_04_a10",  A_10C,  dx=300.0,  dy=200.0,  altitude=380.0, heading=210, cruise_speed_mps=90.0),
            PlaneSpec("plane_04_su25", Su_25T, dx=-260.0, dy=-180.0, altitude=420.0, heading=30,  cruise_speed_mps=95.0),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_07_autumn_morning_overcast_planes",
        weather=WEATHER["autumn_morning_overcast"],
        ships=(
            ShipSpec("ship_07_supply_anchor", Ship_Tilde_Supply, dx=0.0, dy=0.0, heading=100),
        ),
        planes=(
            PlaneSpec("plane_07_mig29", MiG_29A, dx=350.0,  dy=-220.0, altitude=460.0, heading=250, cruise_speed_mps=110.0),
            PlaneSpec("plane_07_a10",   A_10C,   dx=-220.0, dy=180.0,  altitude=380.0, heading=80,  cruise_speed_mps=90.0),
            PlaneSpec("plane_07_su25",  Su_25T,  dx=160.0,  dy=300.0,  altitude=440.0, heading=200, cruise_speed_mps=95.0),
        ),
    ),
    MissionSpec(
        name="caucasus_static_neutral_10_autumn_evening_complex_planes",
        weather=WEATHER["autumn_evening_complex"],
        ships=(
            ShipSpec("ship_10_cargo_anchor", Dry_cargo_ship_1, dx=0.0, dy=0.0, heading=300),
        ),
        planes=(
            PlaneSpec("plane_10_a10",   A_10C,   dx=280.0,  dy=180.0,  altitude=400.0, heading=240, cruise_speed_mps=90.0),
            PlaneSpec("plane_10_mig29", MiG_29A, dx=-300.0, dy=-200.0, altitude=480.0, heading=70,  cruise_speed_mps=105.0),
        ),
    ),
)


# Ordered by the prompt's preferred season/weather sequence.
MISSIONS: Sequence[MissionSpec] = (
    SHIP_MISSIONS[0],         # 01: winter morning clear, ships
    SHIP_MISSIONS[1],         # 02: winter day cloudy, ships
    HELICOPTER_MISSIONS[0],   # 03: spring morning haze, helos
    PLANE_MISSIONS[0],        # 04: spring evening broken, planes
    SHIP_MISSIONS[2],         # 05: summer day clear, ships
    HELICOPTER_MISSIONS[1],   # 06: summer evening warm, helos
    PLANE_MISSIONS[1],        # 07: autumn morning overcast, planes
    SHIP_MISSIONS[3],         # 08: autumn day cloudy, ships
    HELICOPTER_MISSIONS[2],   # 09: spring day mist, helos
    PLANE_MISSIONS[2],        # 10: autumn evening complex, planes
)


# ----------------------------------------------------------------------- #
# pydcs helpers
# ----------------------------------------------------------------------- #


def neutral_country(mission: Mission) -> UnitedNationsPeacekeepers:
    """Ensure the neutral coalition exists and return the UN country."""
    if "neutrals" not in mission.coalition:
        mission.coalition["neutrals"] = Coalition("neutrals")
    country = UnitedNationsPeacekeepers()
    mission.coalition["neutrals"].add_country(country)
    return country


def apply_weather(mission: Mission, weather: WeatherSpec) -> None:
    mission.start_time = weather.date
    mission.weather.name = weather.weather
    mission.weather.clouds_density = weather.clouds_density
    mission.weather.clouds_base = weather.clouds_base
    mission.weather.clouds_thickness = weather.clouds_thickness
    mission.weather.visibility_distance = weather.visibility_distance
    mission.weather.fog_visibility = weather.fog_visibility
    mission.weather.fog_thickness = weather.fog_thickness
    mission.weather.enable_fog = weather.fog_thickness > 0
    mission.weather.wind_at_ground.speed = weather.wind_ground_speed


def add_stationary_ship(
    mission: Mission, country: UnitedNationsPeacekeepers, spec: ShipSpec,
) -> None:
    """Add a ship anchored at SEA_BASE + (dx, dy) with zero speed."""
    abs_x = SEA_BASE_X + spec.dx
    abs_y = SEA_BASE_Y + spec.dy
    group = mission.ship_group(
        country,
        spec.name,
        spec.unit_type,
        Point(abs_x, abs_y, mission.terrain),
        heading=spec.heading,
        group_size=1,
    )
    group.task = None
    group.tasks = []
    group.late_activation = False
    for point in group.points:
        point.speed = 0.0
        point.speed_locked = True
        point.ETA_locked = True
        point.tasks = []


def _flying_group_in_air(
    mission: Mission,
    country: UnitedNationsPeacekeepers,
    name: str,
    unit_type: Type[FlyingType],
    dx: float,
    dy: float,
    altitude: float,
    heading_deg: float,
    speed_mps: float,
) -> None:
    """Common path for both helicopters and slow planes.

    Why this is harder than the airfield case
    ----------------------------------------
    The previous version used ``uncontrolled=True`` + ``speed=0``. Both are
    landmines for in-air spawns:

      * ``uncontrolled=True`` disables the AI, so the unit stops following its
        waypoint, then physics drops it into the sea.
      * ``speed=0`` for a helicopter is interpreted by DCS as "stop and land
        here", not "hover".
      * Without an explicit ``unit.alt`` the spawn altitude is sometimes 0
        because the unit dict only carried altitude on the waypoint, not on
        the unit itself.

    The fix used here:

      1. Keep the AI **controlled**, give it ``maintask = Nothing`` so it
         won't pick targets, fire, or burn fuel toward an aggressive task.
      2. Set a non-zero minimum forward speed (helicopters drift slowly,
         planes cruise low and slow).
      3. Wrap the spawn waypoint in an ``OrbitAction`` (Circle pattern) so the
         AI loiters at the spawn point instead of flying off to infinity.
      4. Set ``unit.alt`` and ``unit.alt_type`` explicitly — the waypoint-only
         altitude was the smoking gun behind units appearing on the water.

    The capture pipeline pauses DCS before every screenshot, so even with the
    aircraft constantly in motion each frame freezes a clean static image.
    """
    abs_x = SEA_BASE_X + dx
    abs_y = SEA_BASE_Y + dy
    group = mission.flight_group_inflight(
        country,
        name,
        unit_type,
        Point(abs_x, abs_y, mission.terrain),
        altitude=int(altitude),
        speed=int(speed_mps),
        maintask=task.Nothing,
        group_size=1,
    )
    group.late_activation = False
    # IMPORTANT: do NOT touch ``group.task`` or ``group.tasks`` here. The
    # constructor already wired the maintask=Nothing pieces into ``group.tasks``
    # — wiping it means the AI for fixed-wing aircraft sees no mission and
    # stops flying (run 4 made planes fall straight into the sea this way).
    # Helicopters tolerated the wipe because hover behaviour is more forgiving
    # than fixed-wing autopilot.
    #
    # Also do NOT set ``uncontrolled = True`` — for an in-air spawn that
    # disables physics-keeping AI and the aircraft falls into the sea.

    for unit in group.units:
        unit.heading = math.radians(heading_deg)
        # Explicit altitude on the unit itself (the waypoint-only altitude
        # was what caused units to spawn on the water in run 3).
        unit.alt = int(altitude)
        unit.alt_type = "BARO"
        try:
            unit.pylons.clear()
        except AttributeError:
            pass

    # Lock all waypoints to the same speed/altitude so the AI doesn't
    # "speed up to catch the next waypoint" and exit the capture zone.
    for point in group.points:
        point.speed = float(speed_mps)
        point.speed_locked = True
        point.ETA_locked = True
        point.alt = int(altitude)
        point.alt_type = "BARO"

    # Loiter at the spawn point. Pattern is chosen by speed:
    #   * helicopters (slow, ~15 m/s) → Circle keeps them in a tight box;
    #   * planes  (~90-110 m/s) → Race-Track gives the AI straight-line legs
    #     between turns so it can maintain altitude. Circle at jet speeds
    #     forces a continuous steep bank that costs more altitude than the
    #     200-300 m AGL spawn has to give — that is why planes were splashing.
    spawn_waypoint = group.points[0]
    spawn_waypoint.tasks = []
    pattern = (
        task.OrbitAction.OrbitPattern.Circle
        if speed_mps < 40.0
        else task.OrbitAction.OrbitPattern.RaceTrack
    )
    spawn_waypoint.add_task(task.OrbitAction(
        altitude=int(altitude),
        speed=int(speed_mps),
        pattern=pattern,
    ))


# Helicopters need *some* forward speed or DCS will treat the spawn as a
# landing — 15 m/s ≈ 54 km/h is a slow, in-air loiter that paused frames
# still capture as if the helo were hovering.
_HELICOPTER_LOITER_SPEED_MPS = 15.0


def add_inflight_helicopter(
    mission: Mission, country: UnitedNationsPeacekeepers, spec: HelicopterSpec,
) -> None:
    """Helicopter in a tight Circle loiter at ``spec.altitude`` AGL.

    DCS will not let a controlled helicopter remain at speed 0 without a
    dedicated Hover task — it tries to land. We use a low forward speed +
    Circle orbit so the helo stays inside the capture zone but never lands.
    """
    _flying_group_in_air(
        mission, country, spec.name, spec.unit_type,
        dx=spec.dx, dy=spec.dy,
        altitude=spec.altitude,
        heading_deg=spec.heading,
        speed_mps=_HELICOPTER_LOITER_SPEED_MPS,
    )


def add_inflight_plane(
    mission: Mission, country: UnitedNationsPeacekeepers, spec: PlaneSpec,
) -> None:
    """Plane in a Circle loiter at ``spec.altitude`` AGL.

    Same approach as helicopters but at the spec's cruise speed (typically
    85-110 m/s, comfortably above stall speed for the unit types we use).
    """
    _flying_group_in_air(
        mission, country, spec.name, spec.unit_type,
        dx=spec.dx, dy=spec.dy,
        altitude=spec.altitude,
        heading_deg=spec.heading,
        speed_mps=spec.cruise_speed_mps,
    )


def build_mission(spec: MissionSpec, output_path: Path) -> Path:
    mission = Mission(Caucasus())
    country = neutral_country(mission)
    apply_weather(mission, spec.weather)

    for ship in spec.ships:
        add_stationary_ship(mission, country, ship)
    for helo in spec.helicopters:
        add_inflight_helicopter(mission, country, helo)
    for plane in spec.planes:
        add_inflight_plane(mission, country, plane)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mission.save(str(output_path))
    return output_path


# ----------------------------------------------------------------------- #
# Manifest
# ----------------------------------------------------------------------- #


def _composition_classes(spec: MissionSpec) -> list[str]:
    classes: list[str] = []
    if spec.ships:
        classes.append("ship")
    if spec.helicopters:
        classes.append("helicopter")
    if spec.planes:
        classes.append("airplane")
    return classes


def write_manifest(output_dir: Path, generated: Sequence[Path]) -> Path:
    manifest = {
        "map": "Caucasus",
        "coalition": "neutral / United Nations Peacekeepers",
        "static_policy": (
            "ships: one zero-speed waypoint; "
            "helicopters: in-air hover (speed=0) at 60-120 m AGL, uncontrolled; "
            "planes: in-air level cruise 80-110 m/s at 200-300 m AGL, uncontrolled"
        ),
        "camera_policy": "15 m ASL, pitch 0, orbit 320-580 m around primary target only",
        "sea_base_xy": [SEA_BASE_X, SEA_BASE_Y],
        "missions": [
            {
                "file": path.name,
                "season":     spec.weather.season,
                "time":       spec.weather.date.strftime("%H:%M"),
                "weather":    spec.weather.weather,
                "visibility": spec.weather.visibility,
                "classes":    _composition_classes(spec),
                "ships":       [s.unit_type.id for s in spec.ships],
                "helicopters": [h.unit_type.id for h in spec.helicopters],
                "planes":      [p.unit_type.id for p in spec.planes],
                "neutral":  True,
                "static":   True,
                "frames_requested": spec.frames,
            }
            for spec, path in zip(MISSIONS, generated)
        ],
    }
    manifest_path = output_dir / "mission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


# ----------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 10 static neutral Caucasus missions")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "missions" / "caucasus_static_neutral_100"),
        help="Directory for generated .miz files",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    generated = [
        build_mission(spec, output_dir / f"{spec.name}.miz")
        for spec in MISSIONS
    ]
    manifest_path = write_manifest(output_dir, generated)
    print("\n".join(str(path) for path in generated))
    print(manifest_path)


if __name__ == "__main__":
    main()
