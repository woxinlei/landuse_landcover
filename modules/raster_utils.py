from osgeo import ogr, gdal

def rasterize_shapefile(shapefile, reference_raster, burn_value):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = driver.Open(shapefile, 0)
    if shp_ds is None:
        print(f"无法打开矢量文件: {shapefile}")
        return None

    layer = shp_ds.GetLayer()
    cols = reference_raster.RasterXSize
    rows = reference_raster.RasterYSize
    transform = reference_raster.GetGeoTransform()
    projection = reference_raster.GetProjection()

    target_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(projection)

    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[burn_value])
    return target_ds.GetRasterBand(1).ReadAsArray()
