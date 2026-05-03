// Google Earth Engine Code Editor script.
// 先将 data/processed/gee_station_months_surface_2014_2024.csv 上传为 GEE asset。
// 输出：landsat_station_monthly_2014_2024.csv

var SAMPLE_ASSET = 'users/YOUR_USERNAME/gee_station_months_surface_2014_2024';
var EXPORT_FOLDER = 'gee_exports';
var EXPORT_NAME = 'landsat_station_monthly_2014_2024';
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
    .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);

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
  return image.addBands([ndvi, ndwi, mndwi, ndti]);
}

var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
var landsat = landsat8.merge(landsat9)
  .filterBounds(samples.geometry())
  .map(maskAndScaleLandsat)
  .map(addIndices);

var bandNames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'ndwi', 'mndwi', 'ndti'];

function extractForMonth(feature) {
  var year = ee.Number.parse(feature.get('year'));
  var month = ee.Number.parse(feature.get('month'));
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');
  var geom = feature.geometry().buffer(BUFFER_METERS);
  var collection = landsat.filterBounds(geom).filterDate(start, end);
  var image = collection.median();

  var stats = image.select(bandNames).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 30,
    maxPixels: 1e8,
    tileScale: 4
  });

  var clearPixelCount = image.select('green').reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geom,
    scale: 30,
    maxPixels: 1e8,
    tileScale: 4
  }).get('green');

  var strictValid = ee.Number(clearPixelCount).gte(MIN_CLEAR_PIXEL_COUNT)
    .and(collection.size().gte(1));

  return feature
    .set(stats)
    .set('valid_count', collection.size())
    .set('clear_pixel_count', clearPixelCount)
    .set('landsat_strict_valid_gee', strictValid)
    .set('landsat_buffer_m', BUFFER_METERS);
}

var out = samples.map(extractForMonth);

Export.table.toDrive({
  collection: out,
  description: EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: [
    'station', 'year', 'month',
    'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
    'ndvi', 'ndwi', 'mndwi', 'ndti',
    'valid_count', 'clear_pixel_count', 'landsat_strict_valid_gee', 'landsat_buffer_m'
  ]
});
