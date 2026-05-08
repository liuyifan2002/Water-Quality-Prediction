// Google Earth Engine Code Editor script
// 用途：按 sample_id 显式导出 94 个站点的 Landsat 月尺度特征
// 先上传 data/processed/gee_station_date_samples.csv 为 GEE Table Asset
// 这样每一行都是一个 station-date sample，导出时直接保留 sample_id，避免站点混用

var SAMPLE_ASSET = 'users/YOUR_USERNAME/gee_station_date_samples';
var EXPORT_FOLDER = 'gee_exports';
var EXPORT_NAME = 'hk_landsat_station_date_features';
var BUFFER_METERS = 150;
var MIN_CLEAR_PIXEL_COUNT = 3;

var samplesRaw = ee.FeatureCollection(SAMPLE_ASSET);

function ensurePointGeometry(feature) {
  var lon = ee.Number.parse(feature.get('lon'));
  var lat = ee.Number.parse(feature.get('lat'));
  return feature.setGeometry(ee.Geometry.Point([lon, lat]));
}

var samples = samplesRaw.map(ensurePointGeometry);

function maskAndScaleLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))       // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0));      // snow

  var saturated = image.select('QA_RADSAT').eq(0);

  var optical = image
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
    .multiply(0.0000275)
    .add(-0.2)
    .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    .toFloat();

  var validReflectance = optical.select(['blue', 'green', 'red', 'nir'])
    .reduce(ee.Reducer.min()).gt(0)
    .and(optical.select(['blue', 'green', 'red', 'nir']).reduce(ee.Reducer.max()).lt(1));

  return optical
    .updateMask(clear)
    .updateMask(saturated)
    .updateMask(validReflectance)
    .copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['nir', 'red']).rename('ndvi');
  var ndwi = image.normalizedDifference(['green', 'nir']).rename('ndwi');
  var mndwi = image.normalizedDifference(['green', 'swir1']).rename('mndwi');
  var ndti = image.normalizedDifference(['red', 'green']).rename('ndti');
  return image.addBands([ndvi, ndwi, mndwi, ndti]).toFloat();
}

var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
var landsat = landsat8.merge(landsat9)
  .filterBounds(samples.geometry())
  .map(maskAndScaleLandsat)
  .map(addIndices);

var bandNames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'mndwi', 'ndti'];

function emptyBandImage(names) {
  return ee.Image.constant(ee.List.repeat(0, names.length))
    .rename(names)
    .toFloat()
    .updateMask(ee.Image.constant(0));
}

function getNumberFromDictOrZero(dict, key) {
  dict = ee.Dictionary(dict);
  return ee.Number(ee.Algorithms.If(dict.contains(key), dict.get(key), 0));
}

function extractForSample(feature) {
  var date = ee.Date(feature.get('date'));
  var year = date.get('year');
  var month = date.get('month');
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');
  var geom = feature.geometry().buffer(BUFFER_METERS);

  var collection = landsat
    .filterBounds(geom)
    .filterDate(start, end);

  var imageCount = collection.size();
  var hasImages = imageCount.gt(0);
  var safeCollection = ee.ImageCollection([emptyBandImage(bandNames)]).merge(collection);
  var image = safeCollection.median();

  var stats = image.select(bandNames).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 30,
    maxPixels: 1e8,
    tileScale: 4
  });

  var clearPixelCount = getNumberFromDictOrZero(
    image.select('green').reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: geom,
      scale: 30,
      maxPixels: 1e8,
      tileScale: 4
    }),
    'green'
  );

  var strictValid = clearPixelCount.gte(MIN_CLEAR_PIXEL_COUNT).and(hasImages);

  var status = ee.String(
    ee.Algorithms.If(
      hasImages,
      ee.Algorithms.If(strictValid, 'success', 'low_clear_pixels'),
      'no_data'
    )
  );

  return feature
    .set(stats)
    .set('landsat_image_count', imageCount)
    .set('landsat_has_images', hasImages)
    .set('landsat_clear_pixel_count', clearPixelCount)
    .set('landsat_strict_valid_gee', strictValid)
    .set('landsat_status', status)
    .set('landsat_buffer_m', BUFFER_METERS)
    .set('landsat_window', 'calendar_month');
}

var out = samples.map(extractForSample);

Export.table.toDrive({
  collection: out,
  description: EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: [
    'sample_id', 'Station', 'date', 'year', 'month', 'lon', 'lat',
    'station_name_en', 'water_zone_en',
    'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    'ndvi', 'ndwi', 'mndwi', 'ndti',
    'landsat_image_count', 'landsat_has_images', 'landsat_clear_pixel_count',
    'landsat_strict_valid_gee', 'landsat_status', 'landsat_buffer_m', 'landsat_window'
  ]
});
