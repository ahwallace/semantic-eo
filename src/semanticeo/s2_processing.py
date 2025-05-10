import json
from typing import Any

import odc.stac
import planetary_computer
import pystac_client
import rasterio
import shapely
from odc.geo.geobox import GeoBox
from pyproj import Transformer


def s2_mosaic(aoi: dict[Any], start_date: str, end_date: str, resolution: int = 100):
    """
    Fetch data for ireland.
    """

    geom_4326 = shapely.from_geojson(json.dumps(aoi))

    transformer_2157 = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    geom_2157 = shapely.transform(
        geom_4326, transformer_2157.transform, interleaved=False
    )
    geobox_2157 = GeoBox.from_bbox(
        geom_2157.bounds, crs="EPSG:2157", resolution=resolution
    )

    stac = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = stac.search(
        intersects=aoi,
        datetime=f"{start_date}/{end_date}",
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 80}},
    )

    items = search.item_collection()

    xx = odc.stac.load(
        items,
        geobox=geobox_2157,
        group_by="solar_day",
        chunks={"time": 6, "y": 4096, "x": 4096},
        bands=["B04", "B03", "B02", "SCL"],
        patch_url=planetary_computer.sign,
    )

    # https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-ClassificationMaskGeneration
    valid_masks = xx.SCL.isin([2, 4, 5, 6, 11])

    # https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html#sentinel-2-level-3-quarterly-mosaics
    mosaic = xx[["B04", "B03", "B02"]].where(valid_masks)
    mosaic = mosaic.to_dataarray()
    mosaic = mosaic / 10000
    mosaic = mosaic.chunk({"variable": 3, "time": -1, "y": 768, "x": 768})
    mosaic = mosaic.quantile(q=0.25, dim=["time"])
    mosaic = (mosaic * 3 * 255).clip(0, 255)
    mosaic = mosaic.fillna(0)
    mosaic = mosaic.astype("uint8")

    return mosaic


def mosaic_to_tif(data, aoi, out_path):
    """
    bounds: (minx, miny, maxx, maxy) in epsg 4326 (wgs84), comverted to epsg 2157 (ITM)
    """

    geom_4326 = shapely.from_geojson(json.dumps(aoi))

    transformer_2157 = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    geom_2157 = shapely.transform(
        geom_4326, transformer_2157.transform, interleaved=False
    )

    _, H, W = data.shape

    transform = rasterio.transform.from_bounds(*geom_2157.bounds, height=H, width=W)

    profile = rasterio.profiles.DefaultGTiffProfile()

    profile.update(
        height=H,
        width=W,
        count=3,
        crs="EPSG:2157",
        transform=transform,
    )

    with rasterio.open(out_path, "w", **profile) as dest:
        dest.write(data)


if __name__ == "__main__":

    import coiled

    # coiled.create_software_environment(
    #     name="semantic-eo-env",
    #     conda={
    #         "channels": ["conda-forge", "defaults"],
    #         "dependencies": [
    #             "python=3.12.9",
    #             "dask",
    #             "coiled",
    #             "xarray",
    #             "gdal",
    #             "rasterio",
    #             "shapely",
    #             "pyproj",
    #             "odc-stac",
    #             "odc-geo",
    #             "planetary-computer",
    #             "pystac-client",
    #         ],
    #     },
    # )

    cluster = coiled.Cluster(
        name="semantic-eo-10m",
        n_workers=5,
        worker_vm_types=["e2-highmem-4", "n4-highmem-4"],
        worker_disk_size="80 GiB",
        region="europe-west2",
        spot_policy="spot_with_fallback",
        software="semantic-eo-env",
    )
    client = cluster.get_client()

    # Ireland bounding box in epsg 4326 (wgs84)
    # bounds_4326 = (-10.8182, 51.2163, -5.0919, 55.7091)
    bounds_4326 = [-10.68234795, 51.36473433, -5.94679566, 55.44704815]
    minx, miny, maxx, maxy = bounds_4326
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],  # bottom left
                [minx, maxy],  # top left
                [maxx, maxy],  # top right
                [maxx, miny],  # bottom right
                [minx, miny],  # bottom left
            ]
        ],
    }

    data = s2_mosaic(aoi, start_date="2024-01-01", end_date="2024-04-01", resolution=10)
    data = data.compute().to_numpy()

    out = "./s2_mosaic_10m_2024_q1.tif"
    mosaic_to_tif(data, aoi, out)

    cluster.shutdown()
