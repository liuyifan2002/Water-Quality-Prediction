"""只读核对原GIS站点层与正式station_meta及84站Universe。"""
import arcpy
import csv
from pathlib import Path

root=Path(r"E:\研究生\codex\水质预测")
station_csv=root/r"数据\处理后数据\正式主表\station_meta.csv"
station_shp=Path(r"E:\研究生\gis\香港经纬度\download_20251010_1507_converted.shp")
with station_csv.open(encoding="utf-8-sig",newline="") as f:
    meta={r["Station"]:(float(r["lon"]),float(r["lat"])) for r in csv.DictReader(f)}
gis={}
with arcpy.da.SearchCursor(str(station_shp),["A_Station","SHAPE@XY"]) as rows:
    for station,xy in rows:
        assert station not in gis,station
        gis[station]=xy
assert len(gis)==len(meta)==94,(len(gis),len(meta))
assert set(gis)==set(meta),(set(gis)-set(meta),set(meta)-set(gis))
errors={s:max(abs(gis[s][0]-meta[s][0]),abs(gis[s][1]-meta[s][1])) for s in meta}
print("GIS_STATION_MATCH",len(meta),"MAX_DEGREE_DIFF",max(errors.values()))
for fam in ("chla","tn","tp"):
    p=root/f"runs/{fam}_family_v2/01_station_universe.csv"
    with p.open(encoding="utf-8-sig",newline="") as f:
        ids={r["station_id"] for r in csv.DictReader(f) if r["common_station"].lower()=="true"}
    print(fam,"COMMON_STATIONS",len(ids),"ALL_IN_GIS",ids<=set(gis))
for s in ("SM18","SM19","PT3","VT11"):
    print("KEY",s,meta[s])
