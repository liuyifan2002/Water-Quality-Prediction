"""ArcGIS Pro 3.1：复用原工程布局，去除不明图形并导出可编辑V3地图。"""
from pathlib import Path
import arcpy

ROOT=Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V3_三项定向修改_20261101")
APRX=ROOT/"图2-1_研究区与监测站_可编辑工程.aprx"
GDB=ROOT/"图2-1_制图图层.gdb"
assert APRX.exists() and GDB.exists()
p=arcpy.mp.ArcGISProject(str(APRX))
assert not p.listMaps("V3香港近岸主图")
main=p.createMap("V3香港近岸主图","MAP")
main.spatialReference=arcpy.SpatialReference(32650)
locator=p.createMap("V3全国区位图","MAP")
locator.spatialReference=arcpy.SpatialReference(4326)

def rgb(r,g,b):return {'RGB':[r,g,b,100]}
def add(m,fc,name,fill=None,outline=None,size=None,width=None):
    layer=m.addDataFromPath(str(GDB/fc));layer.name=name
    s=layer.symbology
    try:
        if fill:s.renderer.symbol.color=rgb(*fill)
        if outline:s.renderer.symbol.outlineColor=rgb(*outline)
        if size is not None:s.renderer.symbol.size=size
        if width is not None:s.renderer.symbol.outlineWidth=width
        layer.symbology=s
    except Exception as e:print("STYLE_WARNING",name,str(e))
    return layer

add(main,"sea_backdrop","海域",(230,241,245),(230,241,245),width=0)
add(main,"land_10m_local","邻近陆地",(238,239,232),(165,180,176),width=0.25)
add(main,"hong_kong_land","香港陆地",(215,230,216),(111,145,131),width=0.35)
other=add(main,"other_10","其余10站（94站网络）",(166,178,184),(65,78,86),size=5.1,width=0.5)
formal=add(main,"formal_84","共同正式84站",(33,96,124),(255,255,255),size=5.3,width=0.65)
key=add(main,"key_stations_4","重点站点",(174,88,65),(255,255,255),size=7.7,width=0.7)
try:
    key.showLabels=True
    label=key.listLabelClasses()[0];label.expression="$feature.A_Station";label.visible=True
except Exception as e:print("KEY_LABEL_WARNING",str(e))
zone=add(main,"zone_name_points","主要海区名称",(0,0,0),(0,0,0),size=0.1)
try:
    zone.showLabels=True
    label=zone.listLabelClasses()[0];label.expression="$feature.ZONE_CN";label.visible=True
except Exception as e:print("ZONE_LABEL_WARNING",str(e))
add(locator,"land_110m_world","陆地",(226,233,225),(150,166,157),width=0.15)
add(locator,"study_extent","香港近岸定位",(241,185,164),(168,66,57),width=1.2)

l=p.listLayouts("布局")[0]
c=l.getDefinition("V3")
c.page.width=180;c.page.height=120
keep={"香港地图","地图框","图例","比例尺","指北针"}
c.elements=[e for e in c.elements if e.name in keep]
l.setDefinition(c)
main_frame=l.listElements("MAPFRAME_ELEMENT","香港地图")[0]
main_frame.map=main
main_frame.elementPositionX=41;main_frame.elementPositionY=4
main_frame.elementWidth=135;main_frame.elementHeight=112
inset=l.listElements("MAPFRAME_ELEMENT","地图框")[0]
inset.map=locator
inset.elementPositionX=4;inset.elementPositionY=83
inset.elementWidth=34;inset.elementHeight=33
main_frame.camera.setExtent(arcpy.Extent(160000,2438000,255000,2512000))
inset.camera.setExtent(arcpy.Extent(72,15,135,55))

legend=l.listElements("LEGEND_ELEMENT")[0]
legend.mapFrame=main_frame
legend.showTitle=False
legend.syncNewLayer=False
legend.syncLayerVisibility=False
for item in list(legend.items):legend.removeItem(item)
for layer in (formal,other,key):legend.addItem(layer)
legend.elementPositionX=4.5;legend.elementPositionY=37
legend.elementWidth=34

scale=l.listElements("MAPSURROUND_ELEMENT","比例尺")[0]
scale.mapFrame=main_frame
scale.elementPositionX=5;scale.elementPositionY=17
scale.elementWidth=31
north=l.listElements("MAPSURROUND_ELEMENT","指北针")[0]
north.mapFrame=main_frame
north.elementPositionX=162;north.elementPositionY=96
north.elementWidth=8

p.save()
l.exportToPNG(str(ROOT/"Figure_2_1.png"),resolution=600)
l.exportToPDF(str(ROOT/"Figure_2_1.pdf"),resolution=300)
l.exportToPNG(str(ROOT/"Figure_2_1_preview.png"),resolution=160)
print("EXPORTED",len(p.listMaps()),"maps",len(l.listElements()),"elements",main_frame.camera.scale)
