// Google Earth Engine Code Editor script.
// 先将 数据/处理后数据/衍生版本/gee_station_months_surface_2014_2024.csv 上传为 GEE asset。
// 输出：viirs_ntl_station_monthly_2014_2024.csv

var SAMPLE_ASSET = 'users/YOUR_USERNAME/gee_station_months_surface_2014_2024';
var EXPORT_FOLDER = 'gee_exports';
var EXPORT_NAME = 'viirs_ntl_station_monthly_2014_2024';
var BUFFER_METERS = 1000;

var samplesRaw = ee.FeatureCollection(SAMPLE_ASSET);

function ensurePointGeometry(feature) {
  var lon = ee.Number.parse(feature.get('lon'));
  var lat = ee.Number.parse(feature.get('lat'));
  return feature.setGeometry(ee.Geometry.Point([lon, lat]));
}

var samples = samplesRaw.map(ensurePointGeometry);

var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .select(['avg_rad', 'cf_cvg']);

function extractForMonth(feature) {
  var year = ee.Number.parse(feature.get('year'));
  var month = ee.Number.parse(feature.get('month'));
  var start = ee.Date.fromYMD(year, month, 1);
  var end = start.advance(1, 'month');
  var geom = feature.geometry().buffer(BUFFER_METERS);
  var collection = viirs.filterDate(start, end).filterBounds(geom);
  var image = collection.mean();

  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 500,
    maxPixels: 1e8,
    tileScale: 4
  });

  return feature
    .set('ntl_avg_rad', stats.get('avg_rad'))
    .set('ntl_cf_cvg', stats.get('cf_cvg'))
    .set('viirs_count', collection.size())
    .set('viirs_buffer_m', BUFFER_METERS);
}

var out = samples.map(extractForMonth);

Export.table.toDrive({
  collection: out,
  description: EXPORT_NAME,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: ['station', 'year', 'month', 'ntl_avg_rad', 'ntl_cf_cvg', 'viirs_count', 'viirs_buffer_m']
});
