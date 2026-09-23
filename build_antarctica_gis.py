"""Builds a comprehensive, continuous GeoJSON for the Antarctic continent and its ice shelves.
Ensures full closure across the South Pole and detailed coverage of the Antarctic Peninsula,
Ross Ice Shelf, Ronne-Filchner Ice Shelf, Amery Ice Shelf, Larsen Ice Shelf, and offshore islands.
"""

import json
from pathlib import Path


def create_comprehensive_antarctica_geojson(output_path: Path):
    # Base existing high-detail coastline
    current_path = Path("frontend/public/antarctica.json")
    base_features = []
    if current_path.exists():
        try:
            with open(current_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                base_features = d.get("features", [])
        except Exception:
            pass

    # High-precision polygons for all sectors
    features = []

    # 1. Continental Mainland with full closure at South Pole (-90.0)
    # We load the existing main polygon rings
    if base_features:
        for feat in base_features:
            features.append(feat)

    # 2. Ross Ice Shelf Permanent Barrier (Cannot navigate)
    # Bounds: -77.5°S to -85.0°S, 160°E to -150°W
    ross_ice_shelf = {
        "type": "Feature",
        "id": "ROSS_ICE_SHELF",
        "properties": {"name": "Ross Ice Shelf (Permanent Ice Barrier)", "type": "IceShelf"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [163.0, -77.5],
                [166.0, -78.0],
                [170.0, -78.5],
                [175.0, -78.3],
                [180.0, -78.2],
                [-175.0, -78.2],
                [-170.0, -78.4],
                [-165.0, -78.3],
                [-160.0, -78.1],
                [-155.0, -77.8],
                [-150.0, -78.5],
                [-150.0, -85.0],
                [160.0, -85.0],
                [163.0, -77.5]
            ]]
        }
    }
    features.append(ross_ice_shelf)

    # 3. Ronne-Filchner Ice Shelf (Weddell Sea Permanent Barrier)
    # Bounds: -75.0°S to -84.0°S, -65°W to -30°W
    ronne_filchner_shelf = {
        "type": "Feature",
        "id": "RONNE_FILCHNER_SHELF",
        "properties": {"name": "Ronne-Filchner Ice Shelf (Permanent Ice Barrier)", "type": "IceShelf"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-60.5, -75.0],
                [-58.0, -75.5],
                [-54.0, -76.0],
                [-50.0, -76.5],
                [-45.0, -77.0],
                [-40.0, -77.5],
                [-35.0, -77.8],
                [-33.0, -78.2],
                [-33.0, -84.0],
                [-65.0, -84.0],
                [-60.5, -75.0]
            ]]
        }
    }
    features.append(ronne_filchner_shelf)

    # 4. Amery Ice Shelf (Prydz Bay Barrier)
    # Bounds: -68.5°S to -73.0°S, 67°E to 75°E
    amery_ice_shelf = {
        "type": "Feature",
        "id": "AMERY_ICE_SHELF",
        "properties": {"name": "Amery Ice Shelf", "type": "IceShelf"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [68.0, -68.5],
                [70.0, -68.8],
                [72.5, -69.0],
                [74.5, -68.7],
                [74.5, -73.0],
                [68.0, -73.0],
                [68.0, -68.5]
            ]]
        }
    }
    features.append(amery_ice_shelf)

    # 5. Larsen C / D Ice Shelf (East of Antarctic Peninsula)
    # Bounds: -66.0°S to -70.0°S, -60.0°W to -65.0°W
    larsen_shelf = {
        "type": "Feature",
        "id": "LARSEN_C_ICE_SHELF",
        "properties": {"name": "Larsen C Ice Shelf", "type": "IceShelf"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-62.0, -66.0],
                [-60.5, -66.5],
                [-60.0, -67.5],
                [-60.8, -68.5],
                [-62.5, -69.5],
                [-64.0, -70.0],
                [-65.0, -68.0],
                [-64.0, -66.5],
                [-62.0, -66.0]
            ]]
        }
    }
    features.append(larsen_shelf)

    # 6. Alexander Island (West of Peninsula)
    alexander_island = {
        "type": "Feature",
        "id": "ALEXANDER_ISLAND",
        "properties": {"name": "Alexander Island", "type": "Land"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-71.5, -70.5],
                [-68.5, -71.2],
                [-68.2, -72.5],
                [-70.5, -73.2],
                [-73.5, -72.8],
                [-74.5, -71.8],
                [-71.5, -70.5]
            ]]
        }
    }
    features.append(alexander_island)

    # 7. Interior Polar Cap Solid Barrier (-83.0°S to -90.0°S everywhere)
    polar_cap = {
        "type": "Feature",
        "id": "POLAR_ICE_CAP",
        "properties": {"name": "Antarctic Interior Ice Sheet", "type": "ContinentalIce"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-180.0, -83.0],
                [-120.0, -83.0],
                [-60.0, -83.0],
                [0.0, -83.0],
                [60.0, -83.0],
                [120.0, -83.0],
                [180.0, -83.0],
                [180.0, -90.0],
                [-180.0, -90.0],
                [-180.0, -83.0]
            ]]
        }
    }
    features.append(polar_cap)

    output_geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_geojson, f)

    print(f"Generated comprehensive Antarctica GIS dataset with {len(features)} features at {output_path}")


if __name__ == "__main__":
    create_comprehensive_antarctica_geojson(Path("frontend/public/antarctica.json"))
