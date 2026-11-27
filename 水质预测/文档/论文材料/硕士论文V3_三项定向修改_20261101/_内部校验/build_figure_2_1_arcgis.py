"""以原ArcGIS工程副本和核对过的94站点重建图2-1；不改原工程。"""
from pathlib import Path
import csv
import arcpy

ROOT=Path(r"E:\研究生\codex\水质预测")
OUT=ROOT/r"文档\论文材料\硕士论文V3_三项定向修改_20261101"
SOURCE=Path(r"C:\Users\刘逸凡\Documents\ArcGIS\毕业论文\地图20251223\地图20251223.aprx")
TARGET=OUT/"图2-1_研究区与监测站_可编辑工程.aprx"
GDB=OUT/"图2-1_制图图层.gdb"
assert SOURCE.exists() and not TARGET.exists() and not GDB.exists()
arcpy.env.overwriteOutput=False
arcpy.management.CreateFileGDB(str(OUT),GDB.name)

station_source=Path(r"E:\研究生\gis\香港经纬度\download_20251010_1507_converted.shp")
hk_source=Path(r"E:\研究生\gis\香港特别行政区\香港特别行政区（（大）\香港特别行政区 (1).shp")
land10=Path(r"E:\研究生\gis\natural_earth_vector\10m_physical\ne_10m_land.shp")
land110=Path(r"E:\研究生\gis\natural_earth_vector\110m_physical\ne_110m_land.shp")
countries110=Path(r"E:\研究生\gis\natural_earth_vector\110m_cultural\ne_110m_admin_0_countries.shp")
for p in (station_source,hk_source,land10,land110,countries110):assert p.exists(),p

def dst(name):return str(GDB/name)
def polygon(coords,sr=4326):
    arr=arcpy.Array([arcpy.Point(x,y) for x,y in coords])
    return arcpy.Polygon(arr,arcpy.SpatialReference(sr))
def rect(x0,y0,x1,y1,sr=4326):
    return polygon([(x0,y0),(x0,y1),(x1,y1),(x1,y0),(x0,y0)],sr)
def one_polygon_fc(name,geom):
    arcpy.management.CreateFeatureclass(str(GDB),name,"POLYGON",spatial_reference=geom.spatialReference)
    with arcpy.da.InsertCursor(dst(name),["SHAPE@"]) as cur:cur.insertRow([geom])
    return dst(name)
def copy_local(source,name,clip_extent=None):
    if clip_extent:
        arcpy.management.MakeFeatureLayer(str(source),"tmp_"+name)
        arcpy.management.SelectLayerByLocation("tmp_"+name,"INTERSECT",clip_extent)
        arcpy.management.CopyFeatures("tmp_"+name,dst(name))
        arcpy.management.Delete("tmp_"+name)
    else:arcpy.management.CopyFeatures(str(source),dst(name))
    return dst(name)

study=one_polygon_fc("study_extent",rect(113.78,22.08,114.53,22.65))
sea=one_polygon_fc("sea_backdrop",rect(113.2,21.6,115.0,23.0))
copy_local(station_source,"monitoring_stations_94")
copy_local(hk_source,"hong_kong_land")
copy_local(land10,"land_10m_local",study)
copy_local(land110,"land_110m_world")
copy_local(countries110,"countries_110m")
with (ROOT/r"runs\chla_family_v2\01_station_universe.csv").open(encoding="utf-8-sig",newline="") as f:
    formal84={r["station_id"] for r in csv.DictReader(f) if r["common_station"].lower()=="true"}
assert len(formal84)==84
arcpy.management.AddField(dst("monitoring_stations_94"),"FAMILY84","SHORT")
with arcpy.da.UpdateCursor(dst("monitoring_stations_94"),["A_Station","FAMILY84"]) as cur:
    for row in cur:
        row[1]=int(row[0] in formal84);cur.updateRow(row)
for name,where in (("formal_84","FAMILY84 = 1"),("other_10","FAMILY84 = 0"),
                   ("key_stations_4","A_Station IN ('SM18','SM19','PT3','VT11')")):
    arcpy.management.MakeFeatureLayer(dst("monitoring_stations_94"),"sel_"+name,where)
    arcpy.management.CopyFeatures("sel_"+name,dst(name))
    arcpy.management.Delete("sel_"+name)
assert [int(arcpy.management.GetCount(dst(x))[0]) for x in ("monitoring_stations_94","formal_84","other_10","key_stations_4")]==[94,84,10,4]

# 来源仅为站点元数据；这些文字点用于海区名称标注，不构造海区边界。
with (ROOT/r"数据\处理后数据\正式主表\station_meta.csv").open(encoding="utf-8-sig",newline="") as f:
    records=list(csv.DictReader(f))
zone_cn={"Deep Bay":"后海湾","Victoria Harbour":"维多利亚港","Southern":"南部海域",
         "Tolo Harbour and Channel":"吐露港及赤门","Mirs Bay":"大鹏湾"}
arcpy.management.CreateFeatureclass(str(GDB),"zone_name_points","POINT",spatial_reference=arcpy.SpatialReference(4326))
arcpy.management.AddField(dst("zone_name_points"),"ZONE_CN","TEXT",field_length=30)
with arcpy.da.InsertCursor(dst("zone_name_points"),["SHAPE@XY","ZONE_CN"]) as cur:
    for z,label in zone_cn.items():
        rr=[r for r in records if r["water_zone_en"]==z]
        assert rr,z
        cur.insertRow([(sum(float(r["lon"]) for r in rr)/len(rr),sum(float(r["lat"]) for r in rr)/len(rr)),label])

src=arcpy.mp.ArcGISProject(str(SOURCE));src.saveACopy(str(TARGET));del src
p=arcpy.mp.ArcGISProject(str(TARGET))
main=p.createMap("V3香港近岸主图","MAP")
main.spatialReference=arcpy.SpatialReference(32650)  # WGS84 / UTM zone 50N，米制比例尺
locator=p.createMap("V3全国区位图","MAP")
locator.spatialReference=arcpy.SpatialReference(4326)

def rgb(r,g,b):return {'RGB':[r,g,b,100]}
def add_layer(m,path,name,fill=None,outline=None,size=None,width=None):
    lyr=m.addDataFromPath(path);lyr.name=name
    sym=lyr.symbology
    try:
        if fill:sym.renderer.symbol.color=rgb(*fill)
        if outline:sym.renderer.symbol.outlineColor=rgb(*outline)
        if size is not None:sym.renderer.symbol.size=size
        if width is not None:sym.renderer.symbol.outlineWidth=width
        lyr.symbology=sym
    except Exception as exc:print("SYMBOLOGY_WARNING",name,exc)
    return lyr

add_layer(main,dst("sea_backdrop"),"海域背景",(232,242,246),(232,242,246),width=0)
add_layer(main,dst("land_10m_local"),"近岸陆地",(239,239,231),(165,182,181),width=0.25)
add_layer(main,dst("hong_kong_land"),"香港陆地",(216,231,217),(116,148,134),width=0.35)
add_layer(main,dst("other_10"),"其余10站",(174,183,187),(68,81,87),size=4.4,width=0.5)
add_layer(main,dst("formal_84"),"三指标共同正式84站",(37,92,119),(255,255,255),size=4.8,width=0.6)
key=add_layer(main,dst("key_stations_4"),"正文重点站点",(179,91,68),(255,255,255),size=7.3,width=0.6)
key.showLabels=True
kl=key.listLabelClasses()[0];kl.expression='$feature.A_Station';kl.visible=True
zones=add_layer(main,dst("zone_name_points"),"海区名称")
zs=zones.symbology;zs.renderer.symbol.color=rgb(0,0,0);zs.renderer.symbol.size=0.1;zones.symbology=zs
zones.showLabels=True
zl=zones.listLabelClasses()[0];zl.expression='$feature.ZONE_CN';zl.visible=True

add_layer(locator,dst("land_110m_world"),"全球陆地",(226,231,225),(154,166,160),width=0.2)
add_layer(locator,dst("study_extent"),"香港近岸定位框",(244,202,177),(177,82,62),width=0.7)

lyt=p.createLayout(180,120,"MILLIMETERS","V3图2-1研究区图")
def page_rect(x0,y0,x1,y1):return rect(x0,y0,x1,y1,0)
mf=lyt.createMapFrame(page_rect(44,5,175,115),main,"香港近岸主图")
mins=[]
with arcpy.da.SearchCursor(dst("monitoring_stations_94"),["SHAPE@"]) as cur:
    for (geom,) in cur:
        q=geom.projectAs(arcpy.SpatialReference(32650)).centroid
        mins.append((q.X,q.Y))
x0=min(x for x,y in mins)-15000;x1=max(x for x,y in mins)+15000
y0=min(y for x,y in mins)-12000;y1=max(y for x,y in mins)+12000
mf.camera.setExtent(arcpy.Extent(x0,y0,x1,y1))
inset=lyt.createMapFrame(page_rect(6,82,40,114),locator,"全国华南沿海定位图")
inset.camera.setExtent(arcpy.Extent(72,15,135,55))

def txt(x,y,s,pt=8,bold=False):
    return p.createTextElement(lyt,arcpy.Point(x,y),"POINT",s,pt,"Microsoft YaHei",
                               "Bold" if bold else "Regular",name="注_"+str(x)+"_"+str(y))
txt(7,77,"全国—华南区位",7.5)
txt(7,70,"● 84站：Chl-a/TN/TP",7.3)
txt(7,64,"● 10站：其余监测点",7.3)
txt(7,58,"● SM18等重点站",7.3)
txt(7,51,"Secchi覆盖全部94站",7.2)
txt(7,40,"WGS84 / UTM 50N",6.7)
txt(7,35,"站点：香港环保署",6.7)
txt(7,30,"底图：Natural Earth",6.7)

try:
    scale_style=p.listStyleItems("ArcGIS 2D","SCALE_BAR","Scale Line 1 Metric")
    sb=lyt.createMapSurroundElement(page_rect(7,14,38,21),"SCALE_BAR",mf,
                                    scale_style[0] if scale_style else None,"主图动态比例尺")
    print("SCALEBAR",sb.name)
except Exception as exc:print("SCALEBAR_WARNING",exc)
try:
    north_style=p.listStyleItems("ArcGIS 2D","NORTH_ARROW","ArcGIS North 1")
    na=lyt.createMapSurroundElement(arcpy.Point(157,96),"NORTH_ARROW",mf,
                                    north_style[0] if north_style else None,"主图指北针")
    na.elementWidth=6
    print("NORTH_ARROW",na.name)
except Exception as exc:print("NORTH_WARNING",exc)

p.save()
png=str(OUT/"Figure_2_1.png")
pdf=str(OUT/"Figure_2_1.pdf")
preview=str(OUT/"Figure_2_1_preview.png")
lyt.exportToPNG(png,resolution=600)
lyt.exportToPDF(pdf,resolution=300)
lyt.exportToPNG(preview,resolution=160)
print("MAP_READY",TARGET,png,pdf,"MAIN_SCALE",mf.camera.scale,"STATIONS",94,"FORMAL",84)
