// Google Earth Engine Code Editor script.
// Upload data/processed/gee_station_date_samples.csv as an asset first.

var SAMPLE_ASSET = 'users/YOUR_USERNAME/gee_station_date_samples';
var EXPORT_FOLDER = 'gee_exports';
var EXPORT_NAME = 'hk_s2_station_date_features';
var LOOKBACK_DAYS = 7;
var BUFFER_METERS = 90;
var CLOUD_PROB_THRESHOLD = 40;
var MIN_VALID_PIXEL_COUNT = 5;

var samplesRaw = ee.FeatureCollection(SAMPLE_ASSET);

function ensurePointGeometry(feature) {
  var lon = ee.Number.parse(feature.get('lon'));
  var lat = ee.Number.parse(feature.get('lat'));
  return feature.setGeometry(ee.Geometry.Point([lon, lat]));
}

var samples = samplesRaw.map(ensurePointGeometry);

function maskS2Clouds(image) {
  var scl = image.select('SCL');
  var valid = scl.neq(0)
    .and(scl.neq(1))
    .and(scl.neq(3))
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  var cloudProb = ee.Image(image.get('cloud_mask')).select('probability');
  var cloudProbMask = cloudProb.lt(CLOUD_PROB_THRESHOLD);
  var qa60 = image.select('QA60');
  var qaClear = qa60.bitwiseAnd(1 << 10).eq(0)
    .and(qa60.bitwiseAnd(1 << 11).eq(0));

  var scaled = image
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11', 'B12'])
    .multiply(0.0001)
    .toFloat();

  var validReflectance = scaled.select(['B2', 'B3', 'B4', 'B8'])
    .reduce(ee.Reducer.min()).gt(0)
    .and(scaled.select(['B2', 'B3', 'B4', 'B8']).reduce(ee.Reducer.max()).lt(1));

  return scaled
    .updateMask(valid)
    .updateMask(cloudProbMask)
    .updateMask(qaClear)
    .updateMask(validReflectance)
    .copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI');
  var turbidityProxy = image.normalizedDifference(['B4', 'B3']).rename('NDTI');
  return image.addBands([ndwi, mndwi, ndci, turbidityProxy]);
}

var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(samples.geometry());

var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterBounds(samples.geometry());

var s2Joined = ee.Join.saveFirst('cloud_mask').apply({
  primary: s2Sr,
  secondary: s2Clouds,
  condition: ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
  })
});

var s2 = ee.ImageCollection(s2Joined)
  .filter(ee.Filter.notNull(['cloud_mask']))
  .map(maskS2Clouds)
  .map(addIndices);

var bandNames = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11', 'B12', 'NDWI', 'MNDWI', 'NDCI', 'NDTI'];

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
  var geom = feature.geometry().buffer(BUFFER_METERS);
  var collection = s2
    .filterBounds(geom)
    .filterDate(date.advance(-LOOKBACK_DAYS, 'day'), date.advance(1, 'day'));

  var imageCount = collection.size();
  var hasImages = imageCount.gt(0);
  var safeCollection = ee.ImageCollection([emptyBandImage(bandNames)]).merge(collection);
  var image = safeCollection.median();
  var stats = image.select(bandNames).reduceRegion({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
    geometry: geom,
    scale: 20,
    maxPixels: 1e8,
    tileScale: 4
  });

  var validPixelCount = getNumberFromDictOrZero(
    image.select('B3').reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: geom,
      scale: 20,
      maxPixels: 1e8,
      tileScale: 4
    }),
    'B3'
  );

  var strictValid = validPixelCount.gte(MIN_VALID_PIXEL_COUNT)
    .and(hasImages);

  return feature
    .set(stats)
    .set('s2_image_count', imageCount)
    .set('s2_has_images', hasImages)
    .set('s2_valid_pixel_count', validPixelCount)
    .set('s2_strict_valid', strictValid)
    .set('s2_cloud_prob_threshold', CLOUD_PROB_THRESHOLD)
    .set('s2_window_days', LOOKBACK_DAYS)
    .set('s2_buffer_m', BUFFER_METERS);
}

var out = samples.map(extractForSample);

Export.table.toDrive({
  collection: out,
  description: EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV'
});
