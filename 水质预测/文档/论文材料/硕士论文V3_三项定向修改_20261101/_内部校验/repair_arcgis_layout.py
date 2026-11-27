"""移除ArcGIS默认在线底图，修复旧布局图例在缩小页面后的渲染。"""
from pathlib import Path
import arcpy

root=Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V3_三项定向修改_20261101")
p=arcpy.mp.ArcGISProject(str(root/"图2-1_研究区与监测站_可编辑工程.aprx"))
for m in p.listMaps():
    if m.name.startswith("V3"):
        for layer in list(m.listLayers()):
            if layer.name=="Topographic":m.removeLayer(layer)
            elif m.name=="V3香港近岸主图":
                layer.name={"共同正式84站":"正式84站","其余10站（94站网络）":"其余10站","重点站点":"重点站"}.get(layer.name,layer.name)
l=p.listLayouts("布局")[0]
c=l.getDefinition("V3")
for e in c.elements:
    if e.name=="图例":
        e.graphicFrame.backgroundSymbol=None
        e.graphicFrame.borderSymbol=None
        e.graphicFrame.shadowSymbol=None
        e.autoAdd=False
        e.showTitle=False
        for item in e.items:
            item.name={"共同正式84站":"正式84站","其余10站（94站网络）":"其余10站","重点站点":"重点站"}.get(item.name,item.name)
            item.isVisible=True
            item.showHeading=False
            item.showDescription=False
            item.showLayerName=False
            item.showLabels=True
    if e.name=="比例尺":
        e.fittingStrategy="AdjustWidth"
        e.division=5
        e.divisions=2
        e.subdivisions=0
        e.labelFrequency="Divisions"
l.setDefinition(c)
l.listElements("LEGEND_ELEMENT")[0].elementPositionY=43
p.save()
l.exportToPNG(str(root/"Figure_2_1.png"),resolution=600)
l.exportToPDF(str(root/"Figure_2_1.pdf"),resolution=300)
l.exportToPNG(str(root/"Figure_2_1_preview.png"),resolution=160)
print("REPAIRED",[(m.name,[x.name for x in m.listLayers()]) for m in p.listMaps() if m.name.startswith('V3')])
