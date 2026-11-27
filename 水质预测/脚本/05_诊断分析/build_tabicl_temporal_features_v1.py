"""Build strictly pre-sample environmental histories from independent daily sources.

This is an isolated Random Strict extension. It never changes the frozen 77
features, sample IDs, target values, or source files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_context_temporal_adaptation_v1'
TABLE = ROOT / '数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv'
RAIN = ROOT / '数据/原始数据/daily_HKO_RF_ALL.csv'
ERA5 = ROOT / '数据/处理后数据/衍生版本/era5_ssrd_sst_daily_2014_2024.csv'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daily_rain() -> pd.DataFrame:
    raw = pd.read_csv(RAIN, skiprows=2, low_memory=False)
    year, month, day, value, completeness = raw.columns[:5]
    date = pd.to_datetime(dict(year=pd.to_numeric(raw[year], errors='coerce'),
                               month=pd.to_numeric(raw[month], errors='coerce'),
                               day=pd.to_numeric(raw[day], errors='coerce')), errors='coerce')
    amount = pd.to_numeric(raw[value], errors='coerce')
    valid = raw[completeness].astype(str).str.strip().eq('C')
    result = pd.DataFrame({'date': date, 'rain_mm': amount.where(valid)})
    result = result.loc[result.date.between('2013-10-01', '2024-12-31')]
    if result.date.duplicated().any():
        raise ValueError('HKO rainfall has duplicate daily dates')
    return result.set_index('date').sort_index()


def daily_era5() -> pd.DataFrame:
    raw = pd.read_csv(ERA5, usecols=['date', 'ssrd_sum_j_m2', 'sst_mean_c'])
    raw['date'] = pd.to_datetime(raw.date, errors='coerce')
    if raw.date.duplicated().any():
        raise ValueError('ERA5 daily source has duplicate dates')
    return raw.set_index('date').sort_index().rename(columns={
        'ssrd_sum_j_m2': 'ssrd_j_m2', 'sst_mean_c': 'sst_c'})


def prior_features(source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    # HKO uses local civil days. ERA5 valid_time was grouped by UTC date in the
    # frozen source script; with sample date but no time, UTC D-1 can extend to
    # 08:00 HKT on sample date D. Therefore ERA5 ends at UTC date D-2.
    calendar = pd.date_range('2014-01-01', '2024-12-31', freq='D')
    aligned = source.reindex(calendar)
    prior = aligned.copy()
    prior['rain_mm'] = aligned['rain_mm'].shift(1)
    prior[['ssrd_j_m2', 'sst_c']] = aligned[['ssrd_j_m2', 'sst_c']].shift(2)
    out = pd.DataFrame(index=calendar)
    specs = []
    for variable, windows, statistic in [
        ('rain_mm', (7, 30, 90), 'sum'),
        ('ssrd_j_m2', (30, 90), 'mean'),
        ('sst_c', (30, 90), 'mean'),
    ]:
        values = prior[variable]
        offset = 1 if variable == 'rain_mm' else 2
        prev_name = f'{variable}_prior{offset}d'
        out[prev_name] = values
        specs.append(dict(feature=prev_name, source='HKO逐日雨量' if variable == 'rain_mm' else 'ERA5逐日汇总',
                          window=(f'采样日前第{offset}个香港本地日' if variable == 'rain_mm'
                                  else f'采样日前第{offset}个UTC日'),
                          aggregation='原日值', unit='mm' if variable == 'rain_mm' else 'J/m²' if variable == 'ssrd_j_m2' else '°C'))
        for window in windows:
            name = f'{variable}_prior{window}d_{statistic}'
            roll = values.rolling(window, min_periods=max(1, int(np.ceil(window * .8))))
            out[name] = roll.sum() if statistic == 'sum' else roll.mean()
            specs.append(dict(feature=name, source='HKO逐日雨量' if variable == 'rain_mm' else 'ERA5逐日汇总',
                              window=f'{window}个自然日，截止采样日前第{offset}个' +
                                  ('香港本地日' if variable == 'rain_mm' else 'UTC日') + '；至少80%有效日',
                              aggregation='累计' if statistic == 'sum' else '均值',
                              unit='mm' if variable == 'rain_mm' else 'J/m²' if variable == 'ssrd_j_m2' else '°C'))
    out.index.name = 'date'
    return out.reset_index(), specs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(TABLE, usecols=['obs_id', 'station', 'date', 'depth'], low_memory=False)
    main['date'] = pd.to_datetime(main.date, errors='coerce')
    if main.obs_id.duplicated().any() or main.date.isna().any():
        raise ValueError('Formal master table has duplicate obs_id or invalid dates')
    sources = daily_rain().join(daily_era5(), how='outer')
    features, specs = prior_features(sources)
    merged = main.merge(features, on='date', how='left', validate='many_to_one')
    names = [item['feature'] for item in specs]
    if len(names) > 15 or any(name in pd.read_csv(TABLE, nrows=0).columns for name in names):
        raise ValueError('Historical feature limit or name collision')
    feature_table = merged[['obs_id', 'station', 'date', 'depth', *names]]
    feature_table.to_parquet(OUT / 'temporal_features_by_obs.parquet', index=False)
    feature_table.to_csv(OUT / 'temporal_features_by_obs.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(specs).assign(
        model_set=lambda df: df.feature.map(lambda name: '仅覆盖审查' if name.startswith('rain_')
            else 'L1' if name.endswith('prior2d') else 'L2'),
        source_sha256=lambda df: df.source.map({
            'HKO逐日雨量': sha(RAIN), 'ERA5逐日汇总': sha(ERA5)})).to_csv(
            OUT / '04_temporal_feature_manifest.csv', index=False, encoding='utf-8-sig')
    coverage = []
    for name in names:
        mask = merged[name].notna()
        coverage.append(dict(feature=name, n_valid=int(mask.sum()), n_total=len(mask),
                             coverage=float(mask.mean()), n_stations=int(merged.loc[mask, 'station'].nunique()),
                             first_date=str(merged.loc[mask, 'date'].min().date()) if mask.any() else '',
                             last_date=str(merged.loc[mask, 'date'].max().date()) if mask.any() else ''))
    pd.DataFrame(coverage).to_csv(OUT / '05_temporal_feature_coverage.csv', index=False, encoding='utf-8-sig')
    inventory = [dict(source='正式主表', path=str(TABLE), rows=len(main),
                      date_min=str(main.date.min().date()), date_max=str(main.date.max().date()),
                      sha256=sha(TABLE)),
                 dict(source='HKO逐日雨量', path=str(RAIN), rows=len(daily_rain()),
                      date_min=str(daily_rain().index.min().date()), date_max=str(daily_rain().index.max().date()),
                      sha256=sha(RAIN)),
                 dict(source='ERA5逐日汇总', path=str(ERA5), rows=len(daily_era5()),
                      date_min=str(daily_era5().index.min().date()), date_max=str(daily_era5().index.max().date()),
                      sha256=sha(ERA5))]
    pd.DataFrame(inventory).to_csv(OUT / '01_temporal_data_inventory.csv', index=False, encoding='utf-8-sig')
    print(f'Built {len(names)} strictly prior-day features for {len(merged)} observations')
    print(pd.DataFrame(coverage)[['feature', 'coverage']].to_string(index=False))


if __name__ == '__main__':
    main()
