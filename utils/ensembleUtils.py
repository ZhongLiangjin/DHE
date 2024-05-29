from collections import defaultdict
from typing import List, Union
import numpy as np
import pandas as pd
import pickle
import json
from matplotlib.ticker import ScalarFormatter
import os
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    def __init__(self, ensembleFolder, flowFile, valDataFile, basinFile, LAIFile):
        """
        :param ensembleFolder: directory of ensemble members.
        :param flowFile: observed runoff csv file.
        :param basinFile: sub-basin shape file.
        :param valDataFile: a pickle file storing remote sensing snow depth and GLEAM ET.
        :param LAIFile: a csv file storing LAI data.
        """

        ensembleDict = defaultdict(dict)
        for file in os.listdir(ensembleFolder):
            if file.startswith('seed'):
                seed = file.split('_')[1]
                with open(os.path.join(ensembleFolder, file, 'config.json'), 'r') as f:
                    self.config = json.load(f)
                # read simulation
                with open(os.path.join(ensembleFolder, file, 'simulation.pkl'), 'rb') as f:
                    simulation = pickle.load(f)

                tRange = pd.date_range(self.config['data']['periods'][1][0], self.config['data']['periods'][2][1])
                for k, v in simulation.items():
                    if k not in ['static', 'dynamic', 'hillSlopeRout', 'riverRout', 'wMul']:
                        ensembleDict[seed][k] = pd.DataFrame(v.T, index=tRange, columns=range(0, 69))
                    else:
                        ensembleDict[seed][k] = v

        seeds = list(ensembleDict.keys())
        variables = ensembleDict[seeds[0]].keys()
        for var in variables:
            if var not in ['static', 'dynamic', 'hillSlopeRout', 'riverRout', 'wMul']:
                mean = np.concatenate([np.expand_dims(ensembleDict[seed][var].values, 0) for seed in seeds],
                                      axis=0).mean(axis=0)
                ensembleDict['mean'][var] = pd.DataFrame(mean, index=tRange, columns=ensembleDict[seeds[0]][var].columns)
            else:
                if var in ['static', 'dynamic']:
                    weighted_paras = []
                    for seed in seeds:
                        # repeat the wMul to match the shape of the parameter
                        if var == 'dynamic':
                            wMul = np.expand_dims(np.expand_dims(ensembleDict[seed]['wMul'], axis=1), axis=-1)
                            wMul = np.tile(wMul, (1, ensembleDict[seed][var].shape[1], 1, ensembleDict[seed][var].shape[-1]))
                        else:
                            wMul = np.expand_dims(ensembleDict[seed]['wMul'], axis=-1)
                            wMul = np.tile(wMul, (1, 1, ensembleDict[seed][var].shape[-1]))
                        # calculate the weighted parameter
                        weighted_paras.append(np.expand_dims((ensembleDict[seed][var] * wMul).sum(axis=-2), axis=0))
                    ensembleDict['mean'][var] = np.concatenate(weighted_paras, axis=0).mean(axis=0)
                else:
                    mean = np.concatenate([np.expand_dims(ensembleDict[seed][var], 0) for seed in seeds],
                                          axis=0).mean(axis=0)
                    ensembleDict['mean'][var] = mean
        self.sim = ensembleDict

        # read streamflow data
        flow = pd.read_csv(flowFile, index_col='DATE', parse_dates=True)
        self.flow = flow[flow.index.isin(tRange)]

        # get validated data including et and snow depth
        with open(valDataFile, 'rb') as f:
            self.valData = pickle.load(f)

        # get LAI data
        self.LAI = pd.read_csv(LAIFile, index_col=0, header=0, parse_dates=True)

        # get basic geometric information of 69 sub-basins.
        self.basins = gpd.read_file(basinFile)

        # get metrics for Q, ET, and snow depth
        self.metric = defaultdict(lambda: defaultdict(dict))
        for key in self.sim.keys():
            self.metric['Q'][key] = self.evalRunoff(tRange='tst', mode=key, scale='d')
            self.metric['ET_gldas'][key] = self.evalET(tRange=['2005-1-1', '2019-12-31'], tScale='m', dataTyp='GLDAS', mode=key)
            self.metric['ET_gleam'][key] = self.evalET(tRange=['2005-1-1', '2019-12-31'], tScale='m', dataTyp='GLEAM', mode=key)
            self.metric['Snow depth'][key] = self.evalSnowDepth(tRange=['2005-1-1', '2019-12-31'], tScale='m', mode=key)
            self.metric['LAI'][key] = self.evalLAI(tRange=['2005-1-1', '2018-12-31'], mode=key, gs_LAI=False)

    def evalRunoff(self, tRange: Union[List[str], str] = 'tst', mode: str = 'mean', scale='d'):
        """
        calculate performance for streamflow.
        :param tRange: plot runoff in the given time range, could be 'tst' or ['2000-1-1', '2012-12-31'].
        :param mode: calculate metrics from ensemble mean or single .
        """
        # get date range
        if isinstance(tRange, str):
            assert tRange in ['trn', 'val', 'tst']
            tRange = self.config['data']['periods'][['trn', 'val', 'tst'].index(tRange)]
            period = pd.date_range(tRange[0], tRange[1])
        else:
            period = pd.date_range(tRange[0], tRange[1])

        # select data from TNH, JUG, MAQ, MET, JIM, and HHY stations
        area = {'HHY': 20930, 'JIM': 45019, 'JUG': 98414, 'MAQ': 86048, 'MET': 59655, 'TNH': 121972}
        dfPred = self.sim[mode]['Qr'][[13, 45, 53, 35, 23, 0]]
        dfPred = dfPred.rename(columns={13: 'HHY', 45: 'JIM', 53: 'MET', 35: 'MAQ', 23: 'JUG', 0: 'TNH'})

        dfPred = dfPred[dfPred.index.isin(period)]
        dfTrue = self.flow[self.flow.index.isin(period)]

        metricDict = defaultdict(dict)
        for i, station in enumerate(list(dfPred.columns)[::-1]):
            unit = 24 * 3600 / (area[station] * 1000)
            if station == 'MET':  # only evaluate the performance in month 5~10
                pred = dfPred[station][(dfPred.index.month >= 5) & (dfPred.index.month <= 10)] * unit
                true = dfTrue[station][(dfTrue.index.month >= 5) & (dfTrue.index.month <= 10)] * unit
            else:
                pred, true = dfPred[station] * unit, dfTrue[station] * unit
            pred.name, true.name = 'pred', 'true'
            dfTemp = pd.concat([pred, true], axis=1)
            dfTemp.dropna(inplace=True)  # drop nan value
            if scale in ['d', 'daily']:
                dfTemp = dfTemp
            elif scale in ['m', 'monthly']:
                dfTemp = dfTemp.groupby([dfTemp.index.year, dfTemp.index.month]).mean()
            else:
                raise ValueError('scale for runoff evaluation must be daily or monthly')
            nse, r, pbias, kge, rmse, nseLog, FLV, FHV = evalFn(true=dfTemp['true'].values, pred=dfTemp['pred'].values)
            dictTemp = {'NSE': nse, 'R': r, 'PBIAS': pbias, 'KGE': kge, 'RMSE': rmse, 'log NSE': nseLog, 'FLV': FLV, 'FHV': FHV}

            if station == 'MET':
                pred, true = dfPred[station] * unit, dfTrue[station] * unit
                pred.name, true.name = 'pred', 'true'
                dfTemp = pd.concat([pred, true], axis=1)
                dfTemp.fillna(0, inplace=True)
                _, _, _, _, _, _, _, FHV = evalFn(true=dfTemp['true'].values, pred=dfTemp['pred'].values)
                dictTemp['FHV'] = FHV
            metricDict[station] = dictTemp

        return metricDict

    def evalSnowDepth(self, tRange: List[str], tScale: str = 'm', mode: str = 'mean'):
        """
        :param tRange: time range with the form of [start time, end time]
        :param tScale: timescale, could be 'd' (abbreviated for daily), 'm' (monthly), and 'a' (annual).
        :param mode: calculate metrics from ensemble mean or single member.
        """
        tRange = pd.date_range(tRange[0], tRange[1])
        obsSdTemp = self.valData['snowDepth'][self.valData['snowDepth'].index.isin(tRange)]
        obsSdTemp.dropna(inplace=True)
        predSdTemp = self.sim[mode]['Sw'][self.sim[mode]['Sw'].index.isin(obsSdTemp.index)]

        # get data of targeted time scale
        if tScale == 'd':
            obsSd, predSd = obsSdTemp, predSdTemp
        elif tScale == 'm':
            obsSd = obsSdTemp.groupby([obsSdTemp.index.year, obsSdTemp.index.month]).mean()
            predSd = predSdTemp.groupby([predSdTemp.index.year, predSdTemp.index.month]).mean()
        elif tScale == 'a':
            obsSd = obsSdTemp.groupby(obsSdTemp.index.year).mean()
            predSd = predSdTemp.groupby(predSdTemp.index.year).mean()
        else:
            raise ValueError('tRange must be d, m, or a')

        dfMetric = pd.DataFrame(columns=['R', 'RMSE'], index=range(0, 69))
        for idx in dfMetric.index:
            obs = obsSd.loc[:, f'basin_{idx}'].values.astype('float')
            pred = predSd.loc[:, idx].values.astype('float')
            dfMetric.loc[idx, 'R'] = np.corrcoef(obs, pred)[0, 1]
            dfMetric.loc[idx, 'RMSE'] = np.sqrt(np.mean((obs - pred) ** 2))

        return {'R': dfMetric['R'].values, 'RMSE': dfMetric['RMSE'].values}

    def evalET(self, tRange: List[str], tScale: str = 'm', dataTyp: str = 'GLEAM', mode: str = 'mean'):
        """
        :param tRange: time range with the form of [start time, end time]
        :param tScale: time scale, could be 'd' (abbreviated for daily), 'm' (monthly), and 'a' (annual).
        :param mode: calculate metrics from ensemble mean or single member.
        """
        assert dataTyp in ['GLEAM', 'GLDAS']
        tRange = pd.date_range(tRange[0], tRange[1])
        obsEtTemp = self.valData[f'{dataTyp}Et'][self.valData[f'{dataTyp}Et'].index.isin(tRange)]
        predEtTemp = self.sim[mode]['E'][self.sim[mode]['E'].index.isin(obsEtTemp.index)]

        # get data of targeted time scale
        if dataTyp == 'GLEAM':
            if tScale == 'd':
                obsEt, predEt = obsEtTemp, predEtTemp
            elif tScale == 'm':
                obsEt = obsEtTemp.groupby([obsEtTemp.index.year, obsEtTemp.index.month]).mean()
                predEt = predEtTemp.groupby([predEtTemp.index.year, predEtTemp.index.month]).mean()
            elif tScale == 'a':
                obsEt = obsEtTemp.groupby(obsEtTemp.index.year).mean()
                predEt = predEtTemp.groupby(predEtTemp.index.year).mean()
            else:
                raise ValueError(f'For {dataTyp} ET, the tRange must be d, m, or a')
        else:
            if tScale == 'm':
                obsEt = obsEtTemp.groupby([obsEtTemp.index.year, obsEtTemp.index.month]).mean()
                predEt = predEtTemp.groupby([predEtTemp.index.year, predEtTemp.index.month]).mean()
            elif tScale == 'a':
                obsEt = obsEtTemp.groupby(obsEtTemp.index.year).mean()
                predEt = predEtTemp.groupby(predEtTemp.index.year).mean()
            else:
                raise ValueError(f'For {dataTyp} ET, the tRange must be m, or a')

        dfMetric = pd.DataFrame(columns=['R', 'RMSE'], index=range(0, 69))
        for idx in dfMetric.index:
            obs = obsEt.loc[:, f'basin_{idx}'].values.astype('float')
            pred = predEt.loc[:, idx].values.astype('float')
            dfMetric.loc[idx, 'R'] = np.corrcoef(obs, pred)[0, 1]
            dfMetric.loc[idx, 'RMSE'] = np.sqrt(np.mean((obs - pred) ** 2))

        return {'R': dfMetric['R'].values, 'RMSE': dfMetric['RMSE'].values}

    def evalLAI(self, tRange: Union[List[str], str] = 'tst', mode: str = 'mean', gs_LAI: bool = False):
        """
        :param tRange: time range with the form of [start time, end time]
        :param mode: calculate metrics from ensemble mean or single member.
        :param gs_LAI: if True, calculate metrics for growing season LAI.
        """
        tRange = pd.date_range(tRange[0], tRange[1])
        df_obs = self.LAI[self.LAI.index.isin(tRange)]
        sim_LAI = self.sim[mode]['LAI'][self.sim[mode]['LAI'].index.isin(tRange)]
        sim_LAI.rename(columns={i: f'basin_{i}' for i in range(0, 69)}, inplace=True)

        # resample simulated LAI to match the observed LAI
        df_sim = pd.DataFrame(index=df_obs.index, columns=df_obs.columns, dtype='float')
        for i in range(len(df_obs.index)):
            if i == len(df_obs.index) - 1:
                start_date = df_obs.index[i]
                df_sim.loc[start_date] = sim_LAI.loc[start_date:, :].mean(axis=0)
            else:
                start_date = df_obs.index[i]
                end_date = df_obs.index[i + 1]
                df_sim.loc[start_date] = sim_LAI.loc[start_date:end_date, :].mean(axis=0)

        # calculate metrics for each sub-basin
        df_metric = pd.DataFrame(index=df_obs.columns, columns=['NSE', 'R', 'RMSE'], dtype='float')
        if gs_LAI:
            df_sim = df_sim[(df_sim.index.month >= 5) & (df_sim.index.month <= 10)]
            df_obs = df_obs[(df_obs.index.month >= 5) & (df_obs.index.month <= 10)]
        else:
            df_sim = df_sim[(df_sim.index.month >= 1) & (df_sim.index.month <= 12)]
            df_obs = df_obs[(df_obs.index.month >= 1) & (df_obs.index.month <= 12)]
        for column in df_obs.columns:
            nse, r, pbias, kge, rmse, nseLog, FLV, FHV = evalFn(true=df_obs.loc[:, column].values,
                                                                pred=df_sim.loc[:, column].values)
            df_metric.loc[column, 'NSE'], df_metric.loc[column, 'R'], df_metric.loc[column, 'RMSE'] = nse, r, rmse
            df_metric.loc[column, 'FLV'], df_metric.loc[column, 'FHV'] = FLV, FHV
        return {'NSE': df_metric['NSE'].values, 'R': df_metric['R'].values, 'RMSE': df_metric['RMSE'].values,
                'FLV': df_metric['FLV'].values, 'FHV': df_metric['FHV'].values}


def evalFn(true: np.ndarray, pred: np.ndarray):
    # calculate nse
    numerator = np.sum((pred - true) ** 2)
    denominator = np.sum((true - np.mean(true)) ** 2)
    nse = 1 - numerator / denominator
    # calculate pearson's correlation coefficient
    r = np.corrcoef(pred, true)[0, 1]
    # calculate the percent bias
    pBias = 100 * (pred - true).sum() / true.sum()
    # calculate KGE
    beta = np.mean(pred) / np.mean(true)
    gamma = (np.std(pred) / np.mean(pred)) / (np.std(true) / np.mean(true))
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    # calculate RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    # calculate log nse
    predLog, trueLog = np.log10(pred + 0.1), np.log10(true + 0.1)
    numerator = np.sum((predLog - trueLog) ** 2)
    denominator = np.sum((trueLog - np.mean(trueLog)) ** 2)
    nseLog = 1 - numerator / denominator
    # FLV the low flows bias bottom 30%, log space
    pred_sort = np.sort(pred)
    target_sort = np.sort(true)
    indexlow = round(0.3 * len(pred_sort))
    lowpred = pred_sort[:indexlow]
    lowtarget = target_sort[:indexlow]
    FLV = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
    # FHV the peak flows bias 2%
    indexhigh = round(0.98 * len(pred_sort))
    highpred = pred_sort[indexhigh:]
    hightarget = target_sort[indexhigh:]
    FHV = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
    return nse, r, pBias, kge, rmse, nseLog, FLV, FHV


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


if __name__ == '__main__':
    model = 'wLAI=0'
    rootDir = '../checkpoints/ensemble'
    evaluator = Evaluator(ensembleFolder=os.path.join(rootDir, model),
                          basinFile='../data/sub-basins/watershed.shp',
                          flowFile='../data/streamflow.csv',
                          LAIFile='../data/LAI.csv',
                          valDataFile='../data/valData.pkl')
