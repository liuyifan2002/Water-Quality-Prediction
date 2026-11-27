// Landsat 高分 patch 导出脚本（本轮仅生成，不在 Codex 中运行）
// 1) 上传 gee_station_date_requests.csv 为 GEE Asset
// 2) 替换 REQUEST_ASSET_ID
// 3) 在 GEE Code Editor 中运行

var REQUEST_ASSET_ID = 'users/your_username/gee_station_date_requests';
var EXPORT_DESC = 'landsat_patch_features_2014_2024';
var PATCH_BUFFER_METERS_3 = 45;  // 约 3x3 @ 30m
var PATCH_BUFFER_METERS_5 = 75;  // 约 5x5 @ 30m
var WINDOW_DAYS = 3;

var requests = ee.FeatureCollection(REQUEST_ASSET_ID);

function prepLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));
  var optical = image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
    .multiply(0.0000275).add(-0.2);
  var lst = ee.Image(ee.Algorithms.If(
    image.bandNames().contains('ST_B10'),
    image.select('ST_B10').multiply(0.00341802).add(149.0).rename('LST_K'),
    ee.Image.constant(-9999).rename('LST_K')
  ));
  optical = optical.addBands(lst);
  optical = optical.updateMask(clear);
  var ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndwi = optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
  var mndwi = optical.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI');
  var redGreenRatio = optical.select('SR_B4').divide(optical.select('SR_B3')).rename('red_green_ratio');
  var nirRedRatio = optical.select('SR_B5').divide(optical.select('SR_B4')).rename('nir_red_ratio');
  return optical.addBands([ndvi, ndwi, mndwi, redGreenRatio, nirRedRatio])
    .copyProperties(image, ['system:time_start', 'LANDSAT_PRODUCT_ID']);
}

function patchStats(image, geom, scale, radiusMeters) {
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
  var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(geom)
    .filterDate(date.advance(-WINDOW_DAYS, 'day'), date.advance(WINDOW_DAYS + 1, 'day'))
    .map(prepLandsat)
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
    date_diff_abs: null
  }, ee.Dictionary({
    request_id: f.get('request_id'),
    station: f.get('station'),
    date: f.get('date'),
    image_id: image.get('LANDSAT_PRODUCT_ID'),
    image_date: image.date().format('YYYY-MM-dd'),
    date_diff_abs: image.get('date_diff_abs')
  })
    .combine(patchStats(image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','LST_K','NDVI','NDWI','MNDWI','red_green_ratio','nir_red_ratio']), geom, 30, PATCH_BUFFER_METERS_3), true)
    .combine(patchStats(image.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','LST_K','NDVI','NDWI','MNDWI','red_green_ratio','nir_red_ratio']), geom, 30, PATCH_BUFFER_METERS_5), true)
  )));
}

var outFc = requests.map(featureForRequest);
Export.table.toDrive({
  collection: outFc,
  description: EXPORT_DESC,
  fileNamePrefix: EXPORT_DESC,
  fileFormat: 'CSV'
});
