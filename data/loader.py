import numpy as np
import pickle
import pandas as pd
from typing import List, Union, Tuple
import torch
from torch.utils.data import dataset, dataloader


class DataLoader:
    def __init__(self, xdataFile: str, flowFile: str, laiFile: str, periods: List[List[str]],
                 spinUp: Union[int, List[int]], seqLen: int, winSize: int, hydStation: Union[str, List[str]],
                 device: Union[str, torch.device] = 'cpu', excludeBsnAttrVar: Union[None, str, List[str]] = None):
        """
        :param xdataFile: path of the data.pkl containing basin mean forcing and attributes, river reach attributes,
            as well as routing order array.
        :param flowFile: path of the streamflow csv file.
        :param laiFile: path of LAI file
        :param periods: a list used to split training, validation, and test periods.
        :param spinUp: a list containing days used to initiate model for training, validation and test sets,
            respectively.
        :param seqLen: sequence length for training set.
        :param winSize: window size to generate sequence.
        :param hydStation: the streamflow at which hydrological stations are selected to train model.
        :param device: whether to use gpu to accelerate.
        :param excludeBsnAttrVar: the excluded variables in basin attributes.
        """
        super().__init__()
        # load input data
        with open(xdataFile, "rb") as f:
            self.xdata = pickle.load(f)
        self.routOrder = self.xdata['routing_order']
        # read streamflow data
        self.flow = pd.read_csv(flowFile, index_col='DATE', parse_dates=True)
        self.LAI = pd.read_csv(laiFile, index_col=0, header=0, parse_dates=True)

        # split training, validation, and test sets
        self.trnPer, self.valPer, self.tstPer = periods
        self.dataAll, self.dataTrn, self.dataVal, self.dataTst = self.splitSets(hydStation, excludeBsnAttrVar)
        self.area = self.dataAll['bsnAttr'][:, self.dataAll['bsnAttrVarLst'].index('area')]

        # calculate mean and std, and do normalization
        self.statDict = calMeanStd(forDataTrn=self.dataTrn['forData'], forVarLst=self.dataTrn['forVarLst'],
                                   bsnAttr=self.dataTrn['bsnAttr'], bsnAttrVarLst=self.dataTrn['bsnAttrVarLst'],
                                   rivAttr=self.dataTrn['rivAttr'], rivAttrVarLst=self.dataTrn['rivAttrVarLst'])
        for dataDict in [self.dataTrn, self.dataVal, self.dataTst]:
            dataDict['forDataNorm'] = self.transNorm(dataDict['forData'], dataDict['forVarLst'], toNorm=True)
            dataDict['bsnAttr'] = self.transNorm(dataDict['bsnAttr'], dataDict['bsnAttrVarLst'], toNorm=True)
            dataDict['rivAttr'] = self.transNorm(dataDict['rivAttr'], dataDict['rivAttrVarLst'], toNorm=True)

        # generate sequences
        self.generateSequence(spinUp=spinUp, winSize=winSize, seqLen=seqLen)

        # get dataset and dataloader
        dsTrn = MyDataset(self.dataTrn, device)
        dsVal = MyDataset(self.dataVal, device)
        dsTst = MyDataset(self.dataTst, device)
        self.loaderTrn = dataloader.DataLoader(dataset=dsTrn, batch_size=1, shuffle=True)
        self.loaderVal = dataloader.DataLoader(dataset=dsVal, batch_size=1, shuffle=False)
        self.loaderTst = dataloader.DataLoader(dataset=dsTst, batch_size=1, shuffle=False)

    def splitSets(self, hydStation: Union[str, List[str]], excludeBsnAttrVar: Union[None, str, List[str]] = None)\
            -> Tuple[dict, dict, dict, dict]:
        # get forData data
        bsns = list(self.xdata['basin_forcing_attr'].keys())
        tRange = list(self.xdata['basin_forcing_attr'][bsns[0]]['forcing'].index)
        forVarLst = list(self.xdata['basin_forcing_attr'][bsns[0]]['forcing'].columns)
        forData = np.concatenate([np.expand_dims(v['forcing'].values, 0) for k, v in
                                  self.xdata['basin_forcing_attr'].items()], axis=0)

        # get basin attributes data
        tempBsnAttrVarLst = list(self.xdata['basin_forcing_attr'][bsns[0]]['attrs'].keys())
        if excludeBsnAttrVar is None:  # No excluded variables
            bsnAttrVarLst = tempBsnAttrVarLst
        else:  # exclude some variables
            if isinstance(excludeBsnAttrVar, str):
                excludeBsnAttrVar = [excludeBsnAttrVar]
            assert isinstance(excludeBsnAttrVar, list)
            bsnAttrVarLst = [var for var in tempBsnAttrVarLst if var not in excludeBsnAttrVar]
        bsnAttr = np.zeros((len(bsns), len(bsnAttrVarLst)))
        for i, bsn in enumerate(bsns):
            bsnAttr[i] = np.array(
                [v for k, v in self.xdata['basin_forcing_attr'][bsn]['attrs'].items() if k in bsnAttrVarLst])

        # get river attributes data
        upIdx = [i for i, bsn in enumerate(bsns) if 'riverAttrs' in self.xdata['basin_forcing_attr'][bsn].keys()]
        rivAttrVarLst = list(self.xdata['basin_forcing_attr'][bsns[upIdx[0]]]['riverAttrs'].keys())
        rivAttr = np.zeros((len(upIdx), len(rivAttrVarLst)))
        for i, idx in enumerate(upIdx):
            rivAttr[i] = np.array([v for k, v in self.xdata['basin_forcing_attr'][bsns[idx]]['riverAttrs'].items()])

        # get runoff data
        if isinstance(hydStation, str):
            hydStation = [hydStation]
        yData = np.concatenate((np.expand_dims(pd.to_numeric(self.flow.index), axis=1),
                                self.flow[hydStation].to_numpy()), axis=1)
        yVarLst = hydStation  # hydrological stations

        # store all info in a dictionary
        dataAll = {'bsns': bsns, 'forTRange': tRange, 'forVarLst': forVarLst, 'forData': forData,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttr,
                   'rivUpIdx': upIdx, 'rivAttrVarLst': rivAttrVarLst, 'rivAttr': rivAttr,
                   'yData': yData, 'yVarLst': yVarLst}

        # split training, validation, and test sets
        tRangeTrn = pd.date_range(self.trnPer[0], self.trnPer[1])
        forIdxTrn = np.arange((tRangeTrn[0] - tRange[0]).days, (tRangeTrn[-1] - tRange[0]).days + 1)
        forDataTrn = forData[:, forIdxTrn]
        # yDataTrn = self.flow[hydStation][self.flow.index.isin(tRangeTrn)].values
        yDataTrn = self.flow[hydStation][self.flow.index.isin(tRangeTrn)]
        yDataTrn = np.concatenate((np.expand_dims(pd.to_numeric(yDataTrn.index), axis=1), yDataTrn.to_numpy()), axis=1)
        dataTrn = {'bsns': bsns, 'tRange': tRangeTrn, 'forVarLst': forVarLst, 'forData': forDataTrn,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttr,
                   'rivUpIdx': upIdx, 'rivAttrVarLst': rivAttrVarLst, 'rivAttr': rivAttr,
                   'yData': yDataTrn, 'yVarLst': yVarLst}

        tRangeVal = pd.date_range(self.valPer[0], self.valPer[1])
        forIdxVal = np.arange((tRangeVal[0] - tRange[0]).days,
                              (tRangeVal[-1] - tRange[0]).days + 1)  # index to get forcing
        forDataVal = forData[:, forIdxVal]
        # yDataVal = self.flow[hydStation][self.flow.index.isin(tRangeVal)].values
        yDataVal = self.flow[hydStation][self.flow.index.isin(tRangeVal)]
        yDataVal = np.concatenate((np.expand_dims(pd.to_numeric(yDataVal.index), axis=1), yDataVal.to_numpy()), axis=1)
        dataVal = {'bsns': bsns, 'tRange': tRangeVal, 'forVarLst': forVarLst, 'forData': forDataVal,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttr,
                   'rivUpIdx': upIdx, 'rivAttrVarLst': rivAttrVarLst, 'rivAttr': rivAttr,
                   'yData': yDataVal, 'yVarLst': yVarLst}

        tRangeTst = pd.date_range(self.tstPer[0], self.tstPer[1])
        forIdxTst = np.arange((tRangeTst[0] - tRange[0]).days, (tRangeTst[-1] - tRange[0]).days + 1)
        forDataTst = forData[:, forIdxTst]
        # yDataTst = self.flow[hydStation][self.flow.index.isin(tRangeTst)].values
        yDataTst = self.flow[hydStation][self.flow.index.isin(tRangeTst)]
        yDataTst = np.concatenate((np.expand_dims(pd.to_numeric(yDataTst.index), axis=1), yDataTst.to_numpy()), axis=1)
        dataTst = {'bsns': bsns, 'tRange': tRangeTst, 'forVarLst': forVarLst, 'forData': forDataTst,
                   'bsnAttrVarLst': bsnAttrVarLst, 'bsnAttr': bsnAttr,
                   'rivUpIdx': upIdx, 'rivAttrVarLst': rivAttrVarLst, 'rivAttr': rivAttr,
                   'yData': yDataTst, 'yVarLst': yVarLst}

        return dataAll, dataTrn, dataVal, dataTst

    def transNorm(self, x: np.ndarray, varLst: Union[str, List[str]], toNorm: bool = True):
        """
        :param x: forcing data, basin attributes or river attributes.
        :param varLst: list or string of variable names.
        :param toNorm: whether to do normalization or reverse normalization.
        """
        if type(varLst) is str:
            varLst = [varLst]
        out = np.zeros(x.shape)
        for k in range(len(varLst)):
            var = varLst[k]
            stat = self.statDict[var]
            if toNorm is True:  # do normalization
                if len(x.shape) == 3:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                        out[:, :, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, :, k] = (x[:, :, k] - stat[0]) / stat[1]
                elif len(x.shape) == 2:
                    if var in ['pr', 'flow']:
                        temp = np.log10(np.sqrt(x[:, k]) + 0.1)
                        out[:, k] = (temp - stat[0]) / stat[1]
                    else:
                        out[:, k] = (x[:, k] - stat[0]) / stat[1]
            else:  # reverse normalization
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        tempTrans = np.power(10, out[:, :, k]) - 0.1
                        tempTrans[tempTrans < 0] = 0  # set negative as zero
                        out[:, :, k] = tempTrans ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[1] + stat[0]
                    if var in ['pr', 'flow']:
                        tempTrans = np.power(10, out[:, k]) - 0.1
                        tempTrans[tempTrans < 0] = 0
                        out[:, k] = tempTrans ** 2
        return out

    def generateSequence(self, spinUp: List[int], winSize: int, seqLen: int):
        shapeTrn = self.dataTrn['forDataNorm'].shape
        numSeq = int((shapeTrn[1] - seqLen) / winSize) + 1
        xTrn = np.zeros((numSeq, shapeTrn[0], seqLen + spinUp[0], shapeTrn[2]))  # forcing to hydrological model
        xnTrn = np.zeros((numSeq, shapeTrn[0], seqLen + spinUp[0], shapeTrn[2]))  # normalized forcing
        yTrn = np.zeros((numSeq, seqLen + spinUp[0], self.dataTrn['yData'].shape[1]))  # streamflow
        idxStart = 0
        for i in range(numSeq):
            if idxStart < spinUp[0]:
                if idxStart == 0:
                    forDataSpin = self.dataTrn['forData'][:, idxStart - spinUp[0]:]
                    forNormDataSpin = self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:]
                    yDataSpin = self.dataTrn['yData'][idxStart - spinUp[0]:]
                else:
                    forDataSpin = np.concatenate((self.dataTrn['forData'][:, idxStart - spinUp[0]:],
                                                  self.dataTrn['forData'][:, :idxStart]), axis=1)
                    forNormDataSpin = np.concatenate((self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:],
                                                      self.dataTrn['forDataNorm'][:, :idxStart]), axis=1)
                    yDataSpin = np.concatenate((self.dataTrn['yData'][idxStart - spinUp[0]:],
                                                self.dataTrn['yData'][:idxStart]), axis=0)
            else:
                forDataSpin = self.dataTrn['forData'][:, idxStart - spinUp[0]:idxStart]
                forNormDataSpin = self.dataTrn['forDataNorm'][:, idxStart - spinUp[0]:idxStart]
                yDataSpin = self.dataTrn['yData'][idxStart - spinUp[0]:idxStart]

            xTrn[i] = np.concatenate((forDataSpin, self.dataTrn['forData'][:, idxStart: idxStart + seqLen]), axis=1)
            xnTrn[i] = np.concatenate((forNormDataSpin, self.dataTrn['forDataNorm'][:, idxStart: idxStart + seqLen]),
                                      axis=1)
            yTrn[i] = np.concatenate((yDataSpin, self.dataTrn['yData'][idxStart: idxStart + seqLen]), axis=0)
            idxStart += winSize
        # validation set
        xValSpin = self.dataTrn['forData'][:, -spinUp[1]:]
        xnValSpin = self.dataTrn['forDataNorm'][:, -spinUp[1]:]
        yValSpin = self.dataTrn['yData'][-spinUp[1]:]
        # test set
        valLen = len(self.dataVal['tRange'])
        if spinUp[2] <= valLen:
            xTstSpin = self.dataVal['forData'][:, -spinUp[2]:]
            xnTstSpin = self.dataVal['forDataNorm'][:, -spinUp[2]:]
            yTstSpin = self.dataVal['yData'][-spinUp[2]:]
        else:
            xTstSpin = np.concatenate((self.dataTrn['forData'][:, -(spinUp[2] - valLen):], self.dataVal['forData']),
                                      axis=1)
            xnTstSpin = np.concatenate((self.dataTrn['forDataNorm'][:, -(spinUp[2] - valLen):],
                                        self.dataVal['forDataNorm']), axis=1)
            yTstSpin = np.concatenate((self.dataTrn['yData'][-(spinUp[2] - valLen):], self.dataVal['yData']),
                                      axis=0)

        # update the training, validation and test sets
        self.dataTrn['forData'] = xTrn
        self.dataTrn['forDataNorm'] = xnTrn
        self.dataTrn['yData'] = yTrn
        self.dataTrn['spinUpLen'] = spinUp[0]

        self.dataVal['forData'] = np.expand_dims(np.concatenate((xValSpin, self.dataVal['forData']), axis=1), axis=0)
        self.dataVal['forDataNorm'] = np.expand_dims(np.concatenate((xnValSpin, self.dataVal['forDataNorm']), axis=1),
                                                     axis=0)
        self.dataVal['yData'] = np.expand_dims(np.concatenate((yValSpin, self.dataVal['yData']), axis=0), axis=0)
        self.dataVal['spinUpLen'] = spinUp[1]

        self.dataTst['forData'] = np.expand_dims(np.concatenate((xTstSpin, self.dataTst['forData']), axis=1), axis=0)
        self.dataTst['forDataNorm'] = np.expand_dims(np.concatenate((xnTstSpin, self.dataTst['forDataNorm']), axis=1),
                                                     axis=0)
        self.dataTst['yData'] = np.expand_dims(np.concatenate((yTstSpin, self.dataTst['yData']), axis=0), axis=0)
        self.dataTst['spinUpLen'] = spinUp[2]


class MyDataset(dataset.Dataset):
    def __init__(self, dataDict: dict, device: Union[str, torch.device]):
        self.x = torch.tensor(dataDict['forData'], device=device, dtype=torch.float32)
        self.xn = torch.tensor(dataDict['forDataNorm'], device=device, dtype=torch.float32)
        self.bsnAttr = torch.tensor(dataDict['bsnAttr'], device=device, dtype=torch.float32)
        self.rivAttr = torch.tensor(dataDict['rivAttr'], device=device, dtype=torch.float32)
        self.y = torch.tensor(dataDict['yData'][:, :, 1:].astype('float'), device=device, dtype=torch.float32)
        self.time = dataDict['yData'][:, :, 0:1]  # timestamp

    def __getitem__(self, index):
        x = self.x[index]
        xn = self.xn[index]
        rivAttr = self.rivAttr
        bsnAttr = self.bsnAttr
        y = self.y[index]
        time = self.time[index]
        return x, xn, bsnAttr, rivAttr, y, time

    def __len__(self):
        return len(self.x)


def calMeanStd(forDataTrn: np.ndarray, forVarLst: List[str], bsnAttr: np.ndarray, bsnAttrVarLst: List[str],
               rivAttr: np.ndarray, rivAttrVarLst: List[str]) -> dict:
    """
    :param forDataTrn: forcing data on the training set with a shape of [N, L, F].
    :param forVarLst: a list of forcing variable names.
    :param bsnAttr: basin attributes data with a shape of [N, F].
    :param bsnAttrVarLst: a list of basin attributes variable names.
    :param rivAttr: river attributes data with a shape of [N, F].
    :param rivAttrVarLst: a list of river attributes variable names.

    :return: a dictionary storing the mean and std for forcing, basin attributes, and river attributes.
    """
    statDict = {}
    # forcing data
    for k, var in enumerate(forVarLst):
        if var in ['pr', 'flow']:
            statDict[var] = calStatGamma(forDataTrn[:, :, k])
        else:
            statDict[var] = calStat(forDataTrn[:, :, k])
    # basin attributes
    for k, var in enumerate(bsnAttrVarLst):
        statDict[var] = calStat(bsnAttr[:, k])
    # river attributes
    for k, var in enumerate(rivAttrVarLst):
        statDict[var] = calStat(rivAttr[:, k])

    return statDict


def calStatGamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics
    # p10 = np.percentile(b, 10).astype(float)
    # p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(np.float32)
    std = np.std(b).astype(np.float32)
    if std < 0.001:
        std = 1
    return [mean, std]


def calStat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    # p10 = np.percentile(b, 10).astype(float)
    # p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(np.float32)
    std = np.std(b).astype(np.float32)
    if std < 0.001:
        std = 1
    return [mean, std]


if __name__ == '__main__':
    loader = DataLoader(xdataFile='./data.pkl', flowFile='./streamflow.csv',
                        laiFile='../../data/LAI/GLASS/AVHRR/lai_sryr.csv',
                        periods=[['1960-1-1', '1989-12-31'], ['1990-1-1', '1999-12-31'], ['2000-1-1', '2019-12-31']],
                        spinUp=[732, 10958, 14610], seqLen=2190, winSize=365, hydStation=['TNH', 'MAQ'])
