"""只读盘点原有ArcGIS工程，不改原工程或图层。"""
import arcpy

paths = [
    r"C:\Users\刘逸凡\Documents\ArcGIS\毕业论文\毕业论文位置\毕业论文位置.aprx",
    r"C:\Users\刘逸凡\Documents\ArcGIS\毕业论文\地图20251223\地图20251223.aprx",
]
for path in paths:
    print("PROJECT", path)
    p = arcpy.mp.ArcGISProject(path)
    for m in p.listMaps():
        print("MAP", m.name, "CRS", m.spatialReference.name, m.spatialReference.factoryCode)
        for lyr in m.listLayers():
            try:
                src = lyr.dataSource
            except Exception:
                src = "<not a feature layer>"
            print("LAYER", lyr.name, "BROKEN", lyr.isBroken, "VISIBLE", lyr.visible, "SOURCE", src)
            if lyr.isFeatureLayer and not lyr.isBroken:
                desc=arcpy.Describe(lyr)
                fields=[f.name for f in arcpy.ListFields(lyr)]
                print("  DATA",desc.shapeType,"COUNT",arcpy.management.GetCount(lyr)[0],"CRS",desc.spatialReference.name,"FIELDS",fields[:20])
                if desc.shapeType == "Point":
                    with arcpy.da.SearchCursor(lyr, fields[:min(3,len(fields))]+["SHAPE@XY"]) as rows:
                        print("  SAMPLE",[r for _,r in zip(range(3),rows)])
    for l in p.listLayouts():
        print("LAYOUT", l.name, "size", l.pageWidth, l.pageHeight)
        for e in l.listElements():
            print("ELEMENT", e.type, e.name)
            if e.type=="MAPFRAME_ELEMENT":
                print("  FRAME",e.map.name,"EXTENT",e.camera.getExtent(),"SCALE",e.camera.scale)
    print("---")
