"""
Utilities for preparing MESH_input_reservoir.txt context data.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import pandas as pd

RESERVOIR_ID_ALIASES = ("COMID", "comid", "main_id", "id", "basin_id")
RESERVOIR_NAME_ALIASES = ("name", "reservoir_name", "res_name", "reservoir")
RESERVOIR_B1_ALIASES = ("b1", "B1", "WF_B1", "wf_b1")
RESERVOIR_B2_ALIASES = ("b2", "B2", "WF_B2", "wf_b2")
RESERVOIR_INPUT_FLAGS = (1, 3)
RESERVOIR_FILE_FORMATS = ("txt", "tb0")
RESERVOIR_FILE_FORMAT_DEFAULT = "txt"
RESERVOIR_NAME_GAP = 25
TB0_FILE_BANNER = "########################################"


def degrees_to_mesh_minutes(degrees: float) -> float:
    """Convert decimal degrees to MESH north-south/east-west minutes."""
    return float(degrees) * 60.0


def normalize_reservoir_file_format(flag: Any) -> tuple[str, bool]:
    """
    Return a supported ``RESERVOIRFILEFLAG`` value.

    Parameters
    ----------
    flag : Any
        Requested reservoir file format.

    Returns
    -------
    tuple[str, bool]
        Normalized format (``txt`` or ``tb0``) and whether the input was
        invalid and corrected to the default.
    """
    if flag is None:
        return RESERVOIR_FILE_FORMAT_DEFAULT, False

    normalized = str(flag).strip().lower()
    if not normalized:
        return RESERVOIR_FILE_FORMAT_DEFAULT, False
    if normalized in RESERVOIR_FILE_FORMATS:
        return normalized, False

    return RESERVOIR_FILE_FORMAT_DEFAULT, True


def default_reservoir_name(basin_id: Union[int, str]) -> str:
    """Return the default reservoir name for a basin without CSV metadata."""
    return str(basin_id)


def mesh_reservoir_name(name: str) -> str:
    """Normalize reservoir names for MESH text files."""
    return " ".join(str(name).split())[:12]


def tb0_coeff(value: float) -> str:
    """Format a reservoir coefficient for ``MESH_input_reservoir.tb0``."""
    value = float(value)
    if value == 0.0:
        return "0.0E+00"
    text = f"{value:.2E}".replace("e", "E")
    if text.endswith("E+00") and len(text) > 4 and text[3] == "0":
        return text[:3] + text[4:]
    return text


def tb0_location(value: float) -> str:
    """Format a decimal-degree location for ``MESH_input_reservoir.tb0``."""
    return f"{float(value):g}"


def tb0_reach_area(value: float) -> str:
    """Format a reach area in square metres for ``MESH_input_reservoir.tb0``."""
    return str(int(round(float(value))))


def mesh_timestep_delta_t_hours(timestep_minutes: float) -> str:
    """Convert MESH ``TIMESTEPFLAG`` minutes to tb0 ``:DeltaT`` hours."""
    hours = float(timestep_minutes) / 60.0
    if hours == int(hours):
        return str(int(hours))
    return f"{hours:g}"


def fortran_i5(value: int) -> str:
    """Format an integer using Fortran ``i5``."""
    return f"{int(value):5d}"


def fortran_f7_1(value: float) -> str:
    """Format a real using Fortran ``f7.1``."""
    return f"{float(value):7.1f}"


def fortran_g10_3(value: float) -> str:
    """Format a real using Fortran ``g10.3``.

    Uses fixed decimals for typical magnitudes and scientific notation (e.g.
    ``3.50E-14``) when values are too small, too large, or would round to zero
    in fixed format.
    """
    value = float(value)
    if value == 0.0:
        return f"{0.0:10.3f}"

    abs_val = abs(value)
    fixed = f"{value:10.3f}"
    if abs_val < 1e-3 or abs_val >= 1e7 or float(fixed) == 0.0:
        return f"{value:10.2E}"
    return fixed


def fortran_a12(value: str) -> str:
    """Format a string using Fortran ``a12``."""
    return f"{str(value)[:12]:12s}"


def fortran_i2(value: int) -> str:
    """Format an integer using Fortran ``i2``."""
    return f"{int(value):2d}"


def format_reservoir_header(n_reservoirs: int) -> str:
    """Return the MESH reservoir file header line."""
    return (
        f"{fortran_i5(n_reservoirs)} {fortran_i5(0)} {fortran_i5(0)}"
    )


def format_reservoir_line(
    lat_min: float,
    lon_min: float,
    b1: float,
    b2: float,
    name: str,
    ireach_num: int,
    location_flag: int = 0,
) -> str:
    """
    Return one aligned, space-separated reservoir record line.

    Kept for tests and direct formatting checks. Production output is rendered
    through ``MESH_input_reservoir.txt.jinja``.
    """
    if location_flag == 1:
        lat_field = fortran_f7_1(lat_min)
        lon_field = fortran_f7_1(lon_min)
    else:
        lat_field = fortran_i5(int(round(lat_min)))
        lon_field = fortran_i5(int(round(lon_min)))

    return (
        f"{lat_field} {lon_field} {fortran_g10_3(b1)} {fortran_g10_3(b2)}"
        f"{(' ' * RESERVOIR_NAME_GAP)}{fortran_a12(mesh_reservoir_name(name))} "
        f"{fortran_i2(ireach_num)}"
    )


def _resolve_column(
    columns: Sequence[str],
    aliases: Sequence[str],
    explicit: Optional[str] = None,
) -> Optional[str]:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Column `{explicit}` not found. Available columns: {list(columns)}"
            )
        return explicit

    for alias in aliases:
        if alias in columns:
            return alias

    return None


def read_reservoir_coefficients(
    csv_path: str,
    main_id: str,
    id_col: Optional[str] = None,
    name_col: Optional[str] = None,
    b1_col: Optional[str] = None,
    b2_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read reservoir power-curve coefficients from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the coefficient CSV file.
    main_id : str
        Name of the basin ID column on the catchment data (e.g. ``COMID``).
    id_col, name_col, b1_col, b2_col : str, optional
        Explicit column names. When omitted, common aliases are detected.
    """
    coeff_df = pd.read_csv(csv_path)
    columns = list(coeff_df.columns)

    resolved_id = _resolve_column(
        columns,
        (main_id, *RESERVOIR_ID_ALIASES),
        explicit=id_col,
    )
    if resolved_id is None:
        raise ValueError(
            f"Could not find a basin ID column in `{csv_path}`. "
            f"Expected one of {(main_id, *RESERVOIR_ID_ALIASES)}."
        )

    resolved_name = _resolve_column(columns, RESERVOIR_NAME_ALIASES, explicit=name_col)
    resolved_b1 = _resolve_column(columns, RESERVOIR_B1_ALIASES, explicit=b1_col)
    resolved_b2 = _resolve_column(columns, RESERVOIR_B2_ALIASES, explicit=b2_col)

    if resolved_b1 is None or resolved_b2 is None:
        raise ValueError(
            f"Could not find B1/B2 coefficient columns in `{csv_path}`. "
            f"Expected columns like {RESERVOIR_B1_ALIASES} and {RESERVOIR_B2_ALIASES}."
        )

    out = coeff_df[[resolved_id, resolved_b1, resolved_b2]].copy()
    out.columns = ["basin_id", "b1", "b2"]
    if resolved_name is not None:
        out["name"] = coeff_df[resolved_name]
    else:
        out["name"] = pd.NA

    out = out.drop_duplicates(subset=["basin_id"], keep="first")
    return out.set_index("basin_id", drop=False)


def _resolve_reach_area(
    lake_area: Any,
    subbasin_area: Any,
    basin_id: Union[int, str],
) -> float:
    """Return lake reach area, preferring ``lake_area`` over subbasin area."""
    if lake_area is not None and not pd.isna(lake_area):
        value = float(lake_area)
        if value > 0.0:
            return value

    if subbasin_area is None or pd.isna(subbasin_area):
        raise ValueError(
            f"Could not resolve reach area for basin `{basin_id}`. "
            "Provide a positive `lake_area` value or subbasin area."
        )

    return float(subbasin_area)


def _lookup_mapping_value(
    mapping: Mapping[Any, Any],
    key: Union[int, str],
) -> Any:
    value = mapping.get(key)
    if value is not None:
        return value

    for map_key, map_value in mapping.items():
        if str(map_key) == str(key):
            return map_value

    return None


def _lookup_coefficient(
    coeff_lookup: Dict[Any, Dict[str, Any]],
    basin_id: Union[int, str],
) -> Optional[Dict[str, Any]]:
    coeff = _lookup_mapping_value(coeff_lookup, basin_id)
    if isinstance(coeff, dict):
        return coeff
    return None


def prepare_reservoir_context(
    cat: gpd.GeoDataFrame,
    coords: pd.DataFrame,
    main_id: str,
    lake_col: str = "IREACH",
    coefficients: Optional[pd.DataFrame] = None,
    location_flag: int = 0,
    use_power_coefficients: bool = True,
    lake_area_col: Optional[str] = None,
    subbasin_areas: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Jinja context for reservoir input files.

    Lakes are taken from ``lake_col`` values greater than zero and written in
    ascending ``IREACH`` order. Locations use catchment centroids converted
    from decimal degrees to MESH minutes (degrees × 60).

    Parameters
    ----------
    cat : geopandas.GeoDataFrame
        Catchment data containing ``main_id`` and ``lake_col``.
    coords : pandas.DataFrame
        Centroid coordinates with ``main_id``, ``lat``, and ``lon`` in decimal
        degrees.
    main_id : str
        Basin identifier column shared by ``cat`` and ``coords``.
    lake_col : str, optional
        Lake numbering column written by ``_compute_ireach`` (default
        ``IREACH``).
    coefficients : pandas.DataFrame, optional
        Coefficient table indexed by basin ID. When a lake is missing from
        this table, ``WF_B1`` and ``WF_B2`` are set to zero.
    location_flag : int, optional
        MESH ``LOCATIONFLAG`` value. ``0`` writes integer ``i5`` locations;
        ``1`` writes real ``f7.1`` locations.
    use_power_coefficients : bool, optional
        When ``False`` (e.g. ``RESERVOIRFLAG`` 3), ``WF_B1`` and ``WF_B2`` are
        always zero. Names may still be taken from ``coefficients`` when
        provided.
    lake_area_col : str, optional
        Catchment column with lake reach area in square metres. When absent or
        invalid for a lake, the corresponding subbasin area is used.
    subbasin_areas : mapping, optional
        Subbasin areas keyed by ``main_id``. Typically taken from
        ``GridArea`` in the drainage database.

    Returns
    -------
    dict
        Context dictionary with keys ``n_reservoirs``, ``location_flag``, and
        ``reservoirs``.
    """
    if location_flag not in (0, 1):
        raise ValueError("`location_flag` must be 0 or 1.")
    if lake_col not in cat.columns:
        raise ValueError(
            f"Lake column `{lake_col}` not found in catchment data. "
            f"Available columns: {list(cat.columns)}"
        )
    if lake_area_col is not None and lake_area_col not in cat.columns:
        raise ValueError(
            f"Lake area column `{lake_area_col}` not found in catchment data. "
            f"Available columns: {list(cat.columns)}"
        )

    lake_columns = [main_id, lake_col]
    if lake_area_col is not None:
        lake_columns.append(lake_area_col)

    lakes = (
        cat.loc[cat[lake_col] > 0, lake_columns]
        .drop_duplicates(subset=[main_id])
        .merge(coords[[main_id, "lat", "lon"]], on=main_id, how="left")
        .sort_values(lake_col)
    )

    coeff_lookup: Dict[Any, Dict[str, Any]] = {}
    if coefficients is not None:
        coeff_lookup = coefficients.set_index("basin_id", drop=False).to_dict("index")

    area_lookup: Dict[Any, Any] = {}
    if subbasin_areas is not None:
        area_lookup = dict(subbasin_areas)
    elif isinstance(cat, gpd.GeoDataFrame) and "geometry" in cat.columns:
        from .geom import _calculate_polygon_areas

        area_gdf = _calculate_polygon_areas(
            cat[[main_id, "geometry"]].copy(),
            target_area_unit="m ** 2",
        )
        area_lookup = dict(
            zip(
                area_gdf[main_id],
                area_gdf["area"].astype(float),
            )
        )

    resolve_reach_area = lake_area_col is not None or bool(area_lookup)

    reservoirs: List[Dict[str, Any]] = []
    for row in lakes.itertuples(index=False):
        basin_id = getattr(row, main_id)
        ireach_num = int(getattr(row, lake_col))
        lat_deg = float(getattr(row, "lat"))
        lon_deg = float(getattr(row, "lon"))
        lat_min = degrees_to_mesh_minutes(lat_deg)
        lon_min = degrees_to_mesh_minutes(lon_deg)
        lake_area = (
            getattr(row, lake_area_col)
            if lake_area_col is not None
            else None
        )
        reach_area = None
        if resolve_reach_area:
            reach_area = _resolve_reach_area(
                lake_area=lake_area,
                subbasin_area=_lookup_mapping_value(area_lookup, basin_id),
                basin_id=basin_id,
            )

        coeff = _lookup_coefficient(coeff_lookup, basin_id)
        if not use_power_coefficients:
            b1 = 0.0
            b2 = 0.0
            if coeff is None:
                name = default_reservoir_name(basin_id)
            else:
                raw_name = coeff.get("name", pd.NA)
                if pd.isna(raw_name) or str(raw_name).strip() == "":
                    name = default_reservoir_name(basin_id)
                else:
                    name = str(raw_name)
        elif coeff is None:
            b1 = 0.0
            b2 = 0.0
            name = default_reservoir_name(basin_id)
        else:
            b1 = float(coeff.get("b1", 0.0) or 0.0)
            b2 = float(coeff.get("b2", 0.0) or 0.0)
            raw_name = coeff.get("name", pd.NA)
            if pd.isna(raw_name) or str(raw_name).strip() == "":
                name = default_reservoir_name(basin_id)
            else:
                name = str(raw_name)

        reservoirs.append(
            {
                "basin_id": basin_id,
                "lat": lat_deg,
                "lon": lon_deg,
                "lat_min": lat_min,
                "lon_min": lon_min,
                "b1": b1,
                "b2": b2,
                "name": name,
                "ireach_num": ireach_num,
                "reach_area": reach_area,
            }
        )

    return {
        "n_reservoirs": len(reservoirs),
        "location_flag": location_flag,
        "reservoirs": reservoirs,
    }


def prepare_reservoir_inflows_context(
    reservoir_context: Dict[str, Any],
    start_time: str,
    delta_t: str,
    meshflow_version: str,
    creation_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Jinja context for ``MESH_input_reservoir.tb0``."""
    missing_areas = [
        reservoir["basin_id"]
        for reservoir in reservoir_context.get("reservoirs", [])
        if reservoir.get("reach_area") is None
    ]
    if missing_areas:
        raise ValueError(
            "Could not resolve reach area for reservoir basins "
            f"{missing_areas}. Map `lake_area` in `ddb_vars` or ensure "
            "subbasin `GridArea` is available."
        )

    return {
        **reservoir_context,
        "start_time": start_time,
        "delta_t": delta_t,
        "meshflow_version": meshflow_version,
        "creation_date": creation_date or date.today().isoformat(),
        "tb0_banner": TB0_FILE_BANNER,
    }
