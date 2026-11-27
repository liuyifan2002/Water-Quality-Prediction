// Sentinel-2 高分 patch 导出脚本（本轮仅生成，不在 Codex 中运行）
// 1) 先将 gee_station_date_requests.csv 上传为 GEE Asset
// 2) 将下面 REQUEST_ASSET_ID 替换为你的实际 Asset ID
// 3) 在 GEE Code Editor 运行导出任务

var REQUEST_ASSET_ID = 'users/your_username/gee_station_date_requests';
var EXPORT_DESC = 's2_patch_features_2014_2024';
var PATCH_BUFFER_METERS_3 = 30;  // 约 3x3 @ 10m
var PATCH_BUFFER_METERS_5 = 50;  // 约 5x5 @ 10m
var WINDOW_DAYS = 3;

var requests = ee.FeatureCollection(REQUEST_ASSET_ID);

function maskS2(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(mask).copyProperties(image, ['system:time_start']);
}

function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI');
  var fai = image.expression(
    'nir - (red + (swir - red) * ((nir_wl - red_wl) / (swir_wl - red_wl)))',
    {
      nir: image.select('B8'),
      red: image.select('B4'),
      swir: image.select('B11'),
      nir_wl: 842,
      red_wl: 665,
      swir_wl: 1610
    }
  ).rename('FAI');
  var redGreenRatio = image.select('B4').divide(image.select('B3')).rename('red_green_ratio');
  return image.addBands([ndvi, ndwi, mndwi, ndci, fai, redGreenRatio]);
}

function patchStats(image, geom, scale, prefix, radiusMeters) {
  var reducer = ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([10, 90]), sharedInputs: true})
    .combine({reducer2: ee.Reducer.count(), sharedInputs: true});
  return image.reduceRegion({
    reducer: reducer,
    geometry: geom.buffer(radiusMeters),
    scale: scale,
    maxPixels: 1e8
  });
}

function featureForRequest(f) {
  var date = ee.Date(f.get('date'));
  var geom = f.geometry();
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(date.advance(-WINDOW_DAYS, 'day'), date.advance(WINDOW_DAYS + 1, 'day'))
    .map(maskS2)
    .map(addIndices)
    .map(function(img) {
      var diff = ee.Number(img.date().difference(date, 'day')).abs();
      return img.set('date_diff_abs', diff);
    })
    .sort('date_diff_abs');

  var image = ee.Image(collection.first());
  var empty = ee.Algorithms.IsEqual(image, null);
  return ee.Feature(geom, ee.Algorithms.If(empty, {
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: null,
    image_date: null,
    date_diff_abs: null,
    cloud_percentage: null
  }, ee.Dictionary({
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: image.get('PRODUCT_ID'),
    image_date: image.date().format('YYYY-MM-dd'),
    date_diff_abs: image.get('date_diff_abs'),
    cloud_percentage: image.get('CLOUDY_PIXEL_PERCENTAGE')
  })
    .combine(patchStats(image.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NDWI','MNDWI','NDCI','FAI','red_green_ratio']), geom, 10, 'p3', PATCH_BUFFER_METERS_3), true)
    .combine(patchStats(image.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NDWI','MNDWI','NDCI','FAI','red_green_ratio']), geom, 10, 'p5', PATCH_BUFFER_METERS_5), true)
  )));
}

var outFc = requests.map(featureForRequest);
Export.table.toDrive({
  collection: outFc,
  description: EXPORT_DESC,
  fileNamePrefix: EXPORT_DESC,
  fileFormat: 'CSV'
});
