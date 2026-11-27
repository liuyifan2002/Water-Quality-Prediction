"""从冻结站点元数据绘制论文图2-1；不接触模型或历史图件。"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from shapely.geometry import box

ROOT = Path(r'E:/研究生/codex/水质预测')
OUT = Path(__file__).resolve().parent
GIS = Path(r'E:/研究生/gis/natural_earth_vector')
stations = pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
assert len(stations) == 94 and stations.Station.is_unique
sets = []
for family in ('chla', 'tn', 'tp'):
    d = pd.read_csv(ROOT/f'transfer/llm_extension_transfer_v1/splits/{family}_family_v2/01_station_universe.csv')
    sets.append(set(d.loc[d.common_station.astype(str).str.lower() == 'true', 'station_id']))
common = set.intersection(*sets)
assert len(common) == 84 and all(x == common for x in sets)
assert set(stations.Station) == set.union(*sets, set(stations.Station))
stations['group'] = stations.Station.map(lambda x: 'common84' if x in common else 'other10')
assert stations.group.value_counts().to_dict() == {'common84':84,'other10':10}
stations[['Station','lon','lat','group']].to_csv(OUT/'图2-1_站点绘图核验.csv', index=False, encoding='utf-8-sig')

extent = (113.78, 114.55, 22.09, 22.67)
poly = box(extent[0], extent[2], extent[1], extent[3])
def layer(rel, bounds=extent):
    g = gpd.read_file(GIS/rel, bbox=(bounds[0],bounds[2],bounds[1],bounds[3]))
    return gpd.clip(g, box(bounds[0],bounds[2],bounds[1],bounds[3]))

land = layer('10m_physical/ne_10m_land.shp')
urban = layer('10m_cultural/ne_10m_urban_areas.shp')
roads = layer('10m_cultural/ne_10m_roads.shp')
hk = gpd.read_file(r'E:/研究生/gis/香港特别行政区/香港特别行政区（（大）/香港特别行政区 (1).shp').clip(poly)
points = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.lon, stations.lat), crs='EPSG:4326')
crs = 'EPSG:32650'
land, urban, roads, hk, points = [g.to_crs(crs) for g in (land,urban,roads,hk,points)]
limits = gpd.GeoSeries([poly],crs='EPSG:4326').to_crs(crs).total_bounds

plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':8.2,'axes.linewidth':.55,'pdf.fonttype':42})
fig = plt.figure(figsize=(7.5,6.4), facecolor='white')
# 主图按真实投影比例铺满版心；定位图叠放在无监测站的西北陆地区。
ax = fig.add_axes([.025,.10,.95,.86])
ax0 = fig.add_axes([.055,.65,.27,.27],zorder=20)
for aa in (ax0,ax):
    aa.set_facecolor('#e7f2f5')
    aa.set_xticks([]); aa.set_yticks([])
    for s in aa.spines.values(): s.set_color('#859aa0'); s.set_linewidth(.55)

# 区域定位图：独立地理层，不使用开题报告旧点位。
inset_bounds=(109.2,117.5,18.4,26.7)
coast = layer('10m_physical/ne_10m_land.shp',inset_bounds)
coast.plot(ax=ax0,facecolor='#dfe6d3',edgecolor='#879a88',linewidth=.35)
ax0.set_xlim(inset_bounds[:2]); ax0.set_ylim(inset_bounds[2:])
ax0.scatter([114.17],[22.32],s=17,facecolor='#245779',edgecolor='white',linewidth=.5,zorder=5)
ax0.add_patch(Rectangle((extent[0],extent[2]),extent[1]-extent[0],extent[3]-extent[2],fill=False,edgecolor='#ba785b',linewidth=1.15,zorder=5))
ax0.text(114.75,22.85,'香港',color='#234457',fontsize=8,fontweight='bold')
ax0.text(109.4,26.3,'华南沿海定位',color='#344e56',fontsize=8.5)
ax0.text(110.1,19.1,'南海',color='#5d8290',fontsize=8)
ax0.annotate('N',xy=(.12,.88),xytext=(.12,.80),xycoords='axes fraction',textcoords='axes fraction',ha='center',fontsize=9,arrowprops=dict(arrowstyle='-|>',lw=.75,color='#334c54'))

# 主图：浅色底图、真实海岸线与道路，两个正式网络层级。
land.plot(ax=ax,facecolor='#e6eadc',edgecolor='#7b968b',linewidth=.58,zorder=1)
urban.plot(ax=ax,facecolor='#e9ddc8',edgecolor='none',alpha=.75,zorder=2)
roads.plot(ax=ax,color='#b5a792',linewidth=.34,alpha=.7,zorder=3)
hk.boundary.plot(ax=ax,color='#637a73',linewidth=.7,zorder=4)
points.loc[points.group=='other10'].plot(ax=ax,color='#9aa5a7',edgecolor='white',linewidth=.4,markersize=23,zorder=6)
points.loc[points.group=='common84'].plot(ax=ax,color='#245779',edgecolor='white',linewidth=.33,markersize=16,zorder=7)
ax.set_xlim(limits[0],limits[2]);ax.set_ylim(limits[1],limits[3]); ax.set_aspect('equal')
# 少量地理名称，不标注单站编号。
for x,y,label in [(113.94,22.59,'深圳'),(114.14,22.27,'香港岛'),(114.02,22.25,'大屿山'),(114.15,22.34,'九龙'),(114.29,22.49,'新界')]:
    xx,yy=gpd.GeoSeries(gpd.points_from_xy([x],[y]),crs='EPSG:4326').to_crs(crs).geometry[0].coords[0]
    ax.text(xx,yy,label,color='#52635b',fontsize=7.3,ha='center',va='center',zorder=8,bbox=dict(facecolor='white',alpha=.55,edgecolor='none',pad=.65))
ax.annotate('N',xy=(.94,.93),xytext=(.94,.83),xycoords='axes fraction',textcoords='axes fraction',ha='center',va='center',fontsize=9,arrowprops=dict(arrowstyle='-|>',lw=.9,color='#294552'))
# UTM米制比例尺，10 km。
x0=limits[0]+.045*(limits[2]-limits[0]);y0=limits[1]+.055*(limits[3]-limits[1]);length=10000
for k in range(2):
    ax.add_patch(Rectangle((x0+k*length/2,y0),length/2,650,facecolor='#344e59' if k==0 else 'white',edgecolor='#344e59',linewidth=.55,zorder=10))
for k in range(3): ax.text(x0+k*length/2,y0+1150,str(k*5),ha='center',fontsize=6.7,color='#344e59')
ax.text(x0+length+1800,y0+1150,'km',fontsize=6.7,color='#344e59')
handles=[Line2D([],[],marker='o',linestyle='',markerfacecolor='#245779',markeredgecolor='white',markersize=6,label='Chl-a、TN、TP共同任务：84站'),Line2D([],[],marker='o',linestyle='',markerfacecolor='#9aa5a7',markeredgecolor='white',markersize=6,label='其余监测站：10站')]
fig.legend(handles=handles,loc='lower right',bbox_to_anchor=(.978,.013),ncol=1,frameon=False,fontsize=7.5)
fig.text(.035,.027,'站点：香港环境保护署；底图：Natural Earth 10m\nSecchi任务覆盖全部94站',fontsize=7.0,color='#51616a')
fig.savefig(OUT/'图2-1_研究区域与94站点.png',dpi=600,facecolor='white')
fig.savefig(OUT/'图2-1_研究区域与94站点.pdf',facecolor='white')
plt.close(fig)
