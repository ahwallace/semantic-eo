from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.mask


def get_image_tiles(mosaic_path, grid_path):

    grid = gpd.read_parquet(grid_path)
    crs = grid.crs.to_epsg()

    fname = Path(mosaic_path).stem
    year = int(fname.split("_")[3])
    quarter = fname.split("_")[4]

    tiles = {}
    with rasterio.open(mosaic_path, "r") as src:
        # Create an empty list for clipped result
        for idx, tile in grid.iterrows():
            output_id = tile.id
            tile_bbox = tile.geometry.bounds

            shape = [tile.geometry]

            # Mask the raster with Polygon
            tile_img, tile_transform = rasterio.mask.mask(src, shape, crop=True)

            tiles[output_id] = {
                "bbox": tile_bbox,
                "crs": f"epsg:{crs}",
                "year": year,
                "quarter": quarter,
                "img_array": tile_img,
                "transform": tile_transform,
            }

    return tiles


if __name__ == "__main__":
    print("Done!")
