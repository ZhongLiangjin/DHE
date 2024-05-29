import os
import numpy as np
import torch
import torch.nn as nn
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import pandas as pd
import logging


class earlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savePath, patience=15, verbose=False, delta=0):
        """
        Args:
            savePath : save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = savePath
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.bestLoss = None
        self.earlyStop = False
        self.delta = delta

    def __call__(self, valLoss, model):
        if self.bestLoss is None:
            self.saveCheckpoint(valLoss, model)
            self.bestLoss = valLoss
        elif valLoss >= self.bestLoss - self.delta:
            self.counter += 1
            logging.info(f'earlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.saveCheckpoint(valLoss, model)
            self.bestLoss = valLoss
            self.counter = 0

    def saveCheckpoint(self, val_loss, model):
        """ Saves model when validation loss decrease. """
        if self.verbose and self.bestLoss is not None:
            logging.info(f'Validation loss decreased ({self.bestLoss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)


class nseLoss(nn.Module):
    def __init__(self, obsLAI: pd.DataFrame, logLossOpt: bool = False, wLog: Union[float, torch.Tensor] = 0.25,
                 wStation: Union[list, None] = None, wLAI: Union[float, torch.Tensor] = 0, cal_gs_LAI: bool = False):
        """
        :param logLossOpt: whether calculate log nse to account for low flows.
        :param wLog: the weight of log nse if logOpt is True.
        :param wStation: the weight of different hydrological stations. Default value of None means equal weights.
        :param wLAI: the weight of LAI.
        :param obsLAI: remotely sensed LAI data.
        :param cal_gs_LAI：whether calculate growing season NSE

        """
        super().__init__()
        self.logOpt = logLossOpt
        self.wLog = wLog
        if isinstance(wLog, float):
            self.wLog = torch.tensor(self.wLog)

        self.wLAI = wLAI
        if isinstance(wLAI, float):
            self.wLAI = torch.tensor(self.wLAI)
        self.obsLAI = obsLAI
        self.cal_gs_LAI = cal_gs_LAI

        self.wStation = wStation

    def cal_nse_Q(self, yTrue: torch.Tensor, yPred: torch.Tensor, spinUp: int):
        """
        :param spinUp: days to spin up model will be excluded when calculating nse.
        :param yTrue: observed streamflow with a shape of [L, N], where L means sequence length and N means the number
                of hydrological stations.
        :param yPred: predicted streamflow with a shape of [L, N].
        :return: nse
        """
        if self.wStation is None:
            wStation = torch.tensor([1]).repeat(yTrue.size(1)).to(yTrue.device)
        else:
            wStation = torch.tensor(self.wStation).to(yTrue.device)

        yTrue, yPred = yTrue[spinUp:], yPred[spinUp:]
        # if the number of nan values exceeds a threshold, the station will be ignored in the later calculation
        idx = torch.tensor(
            [i for i in range(0, yTrue.size(1)) if torch.isnan(yTrue[:, i]).sum() / yTrue.size(0) < 1 / 5])
        idx = idx.to(yTrue.device)
        yTrue, yPred = yTrue.index_select(1, idx), yPred.index_select(1, idx)
        wStation = wStation.index_select(0, idx)
        yTrueMean = torch.nanmean(yTrue, dim=0, keepdim=True)
        # pad 0 for nan values in the observation and then mask the 0 values in later calculation
        mask = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), torch.ones_like(yTrue))
        yTruePad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrue)
        nseTemp = 1 - ((yPred - yTruePad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                  ((yTruePad - yTrueMean) ** 2 * mask).sum(dim=0, keepdim=True)
        nse = (nseTemp * wStation).sum() / wStation.sum()

        if self.logOpt:
            yTrueLog = torch.log10(torch.sqrt(yTrue) + 0.1)
            yPredLog = torch.log10(torch.sqrt(yPred) + 0.1)
            yTrueLogMean = torch.nanmean(yTrueLog, dim=0, keepdim=True)
            yTrueLogPad = torch.where(torch.isnan(yTrue), torch.zeros_like(yTrue), yTrueLog)
            nseLogTemp = 1 - ((yPredLog - yTrueLogPad) ** 2 * mask).sum(dim=0, keepdim=True) / \
                         ((yTrueLogPad - yTrueLogMean) ** 2 * mask).sum(dim=0, keepdim=True)
            nseLog = (nseLogTemp * wStation).sum() / wStation.sum()
            return nse * (1 - self.wLog) + nseLog * self.wLog
        else:
            return nse

    def cal_nse_LAI(self, pred_LAI: torch.Tensor, ts: torch.Tensor, spinUp: int):
        """
        :param pred_LAI: predicted LAI
        :param ts: time stamp of the batch
        :param spinUp: days to spin up model will be excluded when calculating nse.
        """
        pred_LAI, ts = pred_LAI[spinUp:], ts[spinUp:]
        # calculate 8d mean LAI according to given timestamp
        ts = pd.to_datetime(ts.numpy())
        idxLst = torch.tensor(np.where(ts.isin(self.obsLAI.index))[0], device=pred_LAI.device)
        if len(idxLst) > 100:
            mean_pred_LAI = torch.cat([torch.mean(pred_LAI[start:end], dim=0, keepdim=True) for start, end in
                                       zip(idxLst[:-1], idxLst[1:])], dim=0)
            # determine the true LAI and drop the last index
            true_LAI = torch.tensor(self.obsLAI[self.obsLAI.index.isin(ts)].values, device=pred_LAI.device)[:-1]

            if self.cal_gs_LAI:
                # determine the index for growing season
                idxLst_gs = torch.tensor(np.where((ts.isin(self.obsLAI.index)) & (ts.month >= 5) & (ts.month <= 10))[0],
                                         device=pred_LAI.device)
                mask = torch.where(torch.isin(idxLst, idxLst_gs))[0]
                pred_LAI_cal, true_LAI_cal = mean_pred_LAI[mask], true_LAI[mask]

            else:
                pred_LAI_cal, true_LAI_cal = mean_pred_LAI, true_LAI
            true_LAI_cal_mean = torch.nanmean(true_LAI_cal, dim=0, keepdim=True)
            nse = 1 - ((pred_LAI_cal - true_LAI_cal) ** 2).sum(dim=0, keepdim=True) / (
                    (true_LAI_cal - true_LAI_cal_mean) ** 2).sum(dim=0, keepdim=True)

            return nse.mean()
        else:  # for the period without remotely sensed LAI
            return torch.tensor(torch.nan)

    def forward(self, true_Q: torch.Tensor, pred_Q: torch.Tensor, spinUp: int, ts: torch.Tensor,
                pred_LAI: Union[torch.Tensor, None]):
        self.nse_Q = self.cal_nse_Q(true_Q, pred_Q, spinUp)
        self.nse_LAI = torch.tensor(np.nan) if pred_LAI is None else self.cal_nse_LAI(pred_LAI, ts, spinUp)
        if (torch.isnan(self.nse_LAI)) & (~torch.isnan(self.nse_Q)) & (self.wLAI < 1):
            loss = 1 - self.nse_Q
        elif (torch.isnan(self.nse_Q)) & (~torch.isnan(self.nse_LAI)) & (self.wLAI > 0):
            loss = 1 - self.nse_LAI
        else:
            loss = 1 - self.nse_Q * (1 - self.wLAI) - self.nse_LAI * self.wLAI
        return loss


def plotTrnCurve(lossTrnLst: list, lossValLst: list, outPath: str):
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(range(len(lossTrnLst)), lossTrnLst, color=sns.color_palette('tab10')[0], linewidth=1, label='trn')
    ax.plot(range(len(lossValLst)), lossValLst, color=sns.color_palette('tab10')[1], linewidth=1, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outPath)


def saveSimulation(model, loaderVal, loaderTst, outPath, config, obsLAI, cal_gs_LAI, model_type):
    outDict = defaultdict()
    if model_type == 'EXP-BGM':
        outVarDict = dict(zip(['Qr', 'Qs', 'Qb', 'LAI', 'Sw', 'Si', 'Bg', 'E', 'Ew', 'Ei', 'Es', 'Ts', 'Ssl', 'Sss', 'Paras'],
                              ['Qr', 'Qs', 'Qb', 'LAI', 'Swt', 'Sit', 'Bgt', 'E', 'Ew', 'Ei', 'Es', 'Ts', 'Sslt', 'Ssst', 'Paras']))
    elif model_type == 'EXP-HYDRO':
        outVarDict = dict(zip(['Qr', 'Qs', 'Qb', 'Sw', 'Si', 'E', 'Ew', 'Ei', 'Es', 'Ssl', 'Sss', 'Paras'],
                              ['Qr', 'Qs', 'Qb', 'Swt', 'Sit', 'E', 'Ew', 'Ei', 'Es', 'Sslt', 'Ssst', 'Paras']))
    else:
        raise ValueError('The model type must be EXP-HYDRO or EXP-BGM')

    model.eval()
    with torch.no_grad():
        # for validation set
        for x, xn, bsnAttr, rivAttr, y, ts in loaderVal:
            x, xn, y = x.squeeze(0), xn.squeeze(0), y.squeeze(0)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][1], mode='analyse')
            # calculate metrics for streamflow
            pred_Q = output['Qr'][config['train']['outBsnIdx'], :].permute(1, 0).detach().cpu().numpy()
            true_Q = y.detach().cpu().numpy()
            evaluate_Q(true=true_Q, pred=pred_Q, hydroSta=config['data']['hydStations'], mode='val',
                       spinUp=config['data']['spinUp'][1])

            # calculate metrics for LAI
            if model_type == 'EXP-BGM':
                pred_LAI_temp = output['LAI'].permute(1, 0).detach().cpu().numpy()[config['data']['spinUp'][1]:]
                ts = pd.to_datetime(ts.squeeze(0).squeeze(-1).cpu().numpy())[config['data']['spinUp'][1]:]
                true_LAI = obsLAI[obsLAI.index.isin(ts)][:-1]
                idxLst = np.where(ts.isin(obsLAI.index))[0]  # determine the days when remotely sensed LAI are available
                pred_LAI = np.concatenate([np.mean(pred_LAI_temp[start:end], axis=0, keepdims=True) for start, end in
                                           zip(idxLst[:-1], idxLst[1:])], axis=0)
                pred_LAI = pd.DataFrame(pred_LAI, index=true_LAI.index, columns=true_LAI.columns)
                if cal_gs_LAI:
                    # determine growing season LAI
                    true_LAI_gs = true_LAI[(true_LAI.index.month >= 5) & (true_LAI.index.month <= 10)]
                    pred_LAI_gs = pred_LAI[(pred_LAI.index.month >= 5) & (true_LAI.index.month <= 10)]
                    evaluate_LAI(true=true_LAI_gs, pred=pred_LAI_gs)
                else:
                    evaluate_LAI(true=true_LAI, pred=pred_LAI)

            # save the simulated hydrological variables
            for k, v in outVarDict.items():
                if k is not 'Paras':
                    outDict[k] = output[v].detach().cpu().numpy()[:, config['data']['spinUp'][1]:]

        # for test set
        for x, xn, bsnAttr, rivAttr, y, ts in loaderTst:
            x, xn, y, ts = x.squeeze(0), xn.squeeze(0), y.squeeze(0), ts.squeeze(0).squeeze(-1)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][2], mode='analyse')
            # calculate metrics for streamflow
            pred_Q = output['Qr'][config['train']['outBsnIdx'], :].permute(1, 0).detach().cpu().numpy()
            true_Q = y.detach().cpu().numpy()
            evaluate_Q(true=true_Q, pred=pred_Q, hydroSta=config['data']['hydStations'], mode='tst',
                       spinUp=config['data']['spinUp'][2])

            # calculate metrics for LAI
            if model_type == 'EXP-BGM':
                pred_LAI_temp = output['LAI'].permute(1, 0).detach().cpu().numpy()[config['data']['spinUp'][2]:]
                ts = pd.to_datetime(ts.squeeze(0).squeeze(-1).cpu().numpy())[config['data']['spinUp'][2]:]
                true_LAI = obsLAI[obsLAI.index.isin(ts)][:-1]
                idxLst = np.where(ts.isin(obsLAI.index))[0]  # determine the days when remotely sensed LAI are available
                pred_LAI = np.concatenate([np.mean(pred_LAI_temp[start:end], axis=0, keepdims=True) for start, end in
                                           zip(idxLst[:-1], idxLst[1:])], axis=0)
                pred_LAI = pd.DataFrame(pred_LAI, index=true_LAI.index, columns=true_LAI.columns)
                if cal_gs_LAI:
                    # determine growing season LAI
                    true_LAI_gs = true_LAI[(true_LAI.index.month >= 5) & (true_LAI.index.month <= 10)]
                    pred_LAI_gs = pred_LAI[(pred_LAI.index.month >= 5) & (true_LAI.index.month <= 10)]
                    evaluate_LAI(true=true_LAI_gs, pred=pred_LAI_gs)
                else:
                    evaluate_LAI(true=true_LAI, pred=pred_LAI)

            for k, v in outVarDict.items():
                if k is not 'Paras':
                    outTemp = output[v].detach().cpu().numpy()[:, config['data']['spinUp'][2]:]
                    outDict[k] = np.concatenate((outDict[k], outTemp), axis=1)
                else:
                    for paraName, paraValue in output[v].items():
                        outDict[paraName] = paraValue.detach().cpu().numpy()[:, config['data']['spinUp'][2]:] if \
                            paraName is 'dynamic' else paraValue.detach().cpu().numpy()
    with open(os.path.join(config['out'], outPath), 'wb') as f:
        f.write(pickle.dumps(outDict))


def evaluate_Q(true, pred, hydroSta, mode, spinUp):
    logging.info('*' * 100)
    logging.info(f'Calculating metrics on {mode} set')

    for i, station in enumerate(hydroSta):
        trueTemp, predTemp = true[spinUp:, i], pred[spinUp:, i]
        df = pd.DataFrame.from_records([trueTemp, predTemp], index=['true', 'pred']).T
        df.dropna(inplace=True)
        # calculate nse
        numerator = np.sum((df['pred'] - df['true']) ** 2)
        denominator = np.sum((df['true'] - np.mean(df['true'])) ** 2)
        nse = 1 - numerator / denominator
        # calculate pearson's correlation coefficiency
        r = np.corrcoef(df['pred'].values, df['true'].values)[0, 1]
        # calculate PBIAS
        pbias = 100 * (df['pred'] - df['true']).sum() / df['true'].sum()

        logging.info(f'For streamflow at {station} station: NSE={nse:.3f}, R={r:.3f}, PBIAS={pbias:.3f}')


def evaluate_LAI(true, pred):
    # calculate nse
    numerator = np.sum((true.values - pred.values) ** 2, axis=0)
    denominator = np.sum((true.values - np.mean(true.values, axis=0, keepdims=True)) ** 2, axis=0)
    nse = 1 - numerator / denominator
    # calculate pearson's correlation coefficiency
    r = pred.corrwith(true).to_numpy()
    # calculate RMSE
    rmse = np.sqrt(((true - pred) ** 2).mean()).to_numpy()

    logging.info(f'For LAI: mean nse={np.nanmean(nse):.3f}, R={np.nanmean(r):.3f}, RMSE={np.nanmean(rmse):.3f}')
    logging.info(f'For LAI: median nse={np.nanmedian(nse):.3f}, R={np.nanmedian(r):.3f}, RMSE={np.nanmedian(rmse):.3f}')


if __name__ == "__main__":
    def cal_nse_LAI(obsLAI, pred_LAI: torch.Tensor, ts: torch.Tensor, spinUp: int):
        """
        :param pred_LAI: predicted LAI
        :param ts: time stamp of the batch
        :spinUp: days to spin up model will be excluded when calculating nse.
        """
        pred_LAI, ts = pred_LAI[spinUp:], ts[spinUp:]
        # calculate 8d mean LAI according to given timestamp
        ts = pd.to_datetime(ts.numpy())
        idxLst = torch.tensor(np.where(ts.isin(obsLAI.index))[0], device=pred_LAI.device)
        if len(idxLst) > 100:
            mean_pred_LAI = torch.cat([torch.mean(pred_LAI[start:end], dim=0, keepdim=True) for start, end in
                                       zip(idxLst[:-1], idxLst[1:])], dim=0)
            # determine the true LAI and drop the last index
            true_LAI = torch.tensor(obsLAI[obsLAI.index.isin(ts)].values, device=pred_LAI.device)[:-1]

            # determine the index for growing season
            idxLst_gs = torch.tensor(np.where((ts.isin(obsLAI.index)) & (ts.month >= 5) & (ts.month <= 10))[0],
                                     device=pred_LAI.device)
            mask = torch.where(torch.isin(idxLst, idxLst_gs))[0]

            # calculate nse for LAI in growing season
            pred_LAI_gs, true_LAI_gs = mean_pred_LAI[mask], true_LAI[mask]
            true_LAI_gs_mean = torch.nanmean(true_LAI_gs, dim=0, keepdim=True)
            nse = 1 - ((pred_LAI_gs - true_LAI_gs) ** 2).sum(dim=0, keepdim=True) / (
                        (true_LAI_gs - true_LAI_gs_mean) ** 2).sum(dim=0, keepdim=True)
            return nse.mean()
        else:  # for the period without remotely sensed LAI
            return torch.tensor(torch.nan)

    obsLAI = pd.read_csv('../data/LAI.csv', index_col=0, header=0, parse_dates=True)
    ts = torch.tensor(pd.to_numeric(pd.date_range('1999-1-1', '2009-12-31')))
    pred_LAI = pd.DataFrame(index=pd.to_datetime(ts.cpu().numpy()), columns=obsLAI.columns, dtype='float')
    for idx in pred_LAI.index:
        if idx in obsLAI.index:
            pred_LAI.loc[idx, :] = obsLAI.loc[idx, :]
    pred_LAI.interpolate(inplace=True)
    cal_nse_LAI(obsLAI, torch.tensor(pred_LAI.values), ts, 30)