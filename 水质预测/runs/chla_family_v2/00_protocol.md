# CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT

- status: **SPLIT_APPROVED**
- generated: `2026-09-22T14:13:46`
- strict features: `77`
- stations: surface=94, middle=84, bottom=91, common=84
- seed: `20260921`; sklearn: `1.6.1`; Python: `3.11.9`
- one station-level KFold call on sorted common stations; all depths reuse the canonical map.

```text
        task   depth   fold  train_rows  test_rows  train_stations  test_stations  overlap status
chla_surface surface fold_1        7744       2039              67             17        0   PASS
chla_surface surface fold_2        7740       2043              67             17        0   PASS
chla_surface surface fold_3        7923       1860              67             17        0   PASS
chla_surface surface fold_4        7796       1987              67             17        0   PASS
chla_surface surface fold_5        7929       1854              68             16        0   PASS
 chla_middle  middle fold_1        7417       2039              67             17        0   PASS
 chla_middle  middle fold_2        7478       1978              67             17        0   PASS
 chla_middle  middle fold_3        7725       1731              67             17        0   PASS
 chla_middle  middle fold_4        7534       1922              67             17        0   PASS
 chla_middle  middle fold_5        7670       1786              68             16        0   PASS
 chla_bottom  bottom fold_1        7744       2039              67             17        0   PASS
 chla_bottom  bottom fold_2        7740       2043              67             17        0   PASS
 chla_bottom  bottom fold_3        7923       1860              67             17        0   PASS
 chla_bottom  bottom fold_4        7796       1987              67             17        0   PASS
 chla_bottom  bottom fold_5        7929       1854              68             16        0   PASS
```
