import time
import torch
import torch.backends.cudnn as cudnn
from utils.trainTools import earlyStopping, nseLoss, plotTrnCurve, saveSimulation
from pathlib import Path
import os
import json
import argparse
from typing import Union
from collections import defaultdict
from data.loader import DataLoader
from model.ExpBGM import Net
import random
import numpy as np
import logging


def trainModel(model, loaderTrn, loaderVal, lossFn, optimizer, scheduler, earlyStop, config):
    def train(model, loaderTrn, lossFn, optimizer, scheduler, config):
        model.train()
        totalLoss, nse_Q, nse_LAI = 0.0, [], []
        loss_nan = 0
        for i, (x, xn, bsnAttr, rivAttr, y, ts) in enumerate(loaderTrn):
            x, xn, y, ts = x.squeeze(0), xn.squeeze(0), y.squeeze(0), ts.squeeze(0).squeeze(-1)
            bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
            output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][0])
            pred_LAI = output['LAI'].clone().permute(1, 0) if model_type == 'EXP-BGM' else None
            pred_Q = output['Qr'][config['train']['outBsnIdx'], :].clone().permute(1, 0)
            loss = lossFn(true_Q=y, pred_Q=pred_Q, spinUp=config['data']['spinUp'][0], pred_LAI=pred_LAI, ts=ts)
            if ~torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip'])
                optimizer.step()
                totalLoss += loss.item()
            else:
                loss_nan += 1
            nse_Q.append(lossFn.nse_Q)
            nse_LAI.append(lossFn.nse_LAI)
            if i % 2 == 0:
                logging.info(f'Iter {i} of {len(loaderTrn)} Loss: {loss.item():.3f}, streamflow NSE: {lossFn.nse_Q:.3f}'
                             f', LAI NSE: {lossFn.nse_LAI:.3f}')

        epochLoss = totalLoss / (len(loaderTrn) - loss_nan)
        epochNSEQ, epochNSELAI = torch.nanmean(torch.tensor(nse_Q)), torch.nanmean(torch.tensor(nse_LAI))
        if scheduler is not None:
            scheduler.step(epochLoss)
        return epochLoss, epochNSEQ, epochNSELAI

    def valid(model, loaderVal, lossFn, config):
        model.eval()
        with torch.no_grad():
            totalLoss, nse_Q, nse_LAI = 0.0, [], []
            for x, xn, bsnAttr, rivAttr, y, ts in loaderVal:
                x, xn, y, ts = x.squeeze(0), xn.squeeze(0), y.squeeze(0), ts.squeeze(0).squeeze(-1)
                bsnAttr, rivAttr = bsnAttr.squeeze(0), rivAttr.squeeze(0)
                output = model(x, xn, bsnAttr, rivAttr, staIdx=config['data']['spinUp'][1])
                pred_LAI = output['LAI'].clone().permute(1, 0) if model_type == 'EXP-BGM' else None
                pred_Q = output['Qr'][config['train']['outBsnIdx'], :].clone().permute(1, 0)
                loss = lossFn(true_Q=y, pred_Q=pred_Q, spinUp=config['data']['spinUp'][1], pred_LAI=pred_LAI, ts=ts)
                totalLoss += loss.item()
                nse_Q.append(lossFn.nse_Q)
                nse_LAI.append(lossFn.nse_LAI)
            epochLoss = totalLoss / len(loaderVal)
            epochNSEQ, epochNSELAI = torch.nanmean(torch.tensor(nse_Q)), torch.nanmean(torch.tensor(nse_LAI))
        return epochLoss, epochNSEQ, epochNSELAI

    lossTrnLst, lossValLst = [], []
    for epoch in range(config['train']['epochs']):
        logging.info('*' * 100)
        logging.info('Epoch:{:d}/{:d}'.format(epoch, config['train']['epochs']))
        lossTrn, nseQTrn, nseLAITrn = train(model, loaderTrn, lossFn, optimizer, scheduler, config)
        lossTrnLst.append(lossTrn)
        logging.info(f'Epoch training loss: {lossTrn:.3f}, streamflow NSE: {nseQTrn: .3f}, LAI NSE: {nseLAITrn: .3f}')
        lossVal, nseQVal, nseLAIVal = valid(model, loaderVal, lossFn, config)
        lossValLst.append(lossVal)
        logging.info(f'Epoch validation loss: {lossVal:.3f}, streamflow NSE: {nseQVal: .3f}, LAI NSE: {nseLAIVal: .3f}')
        earlyStop(lossVal, model)
        if earlyStop.earlyStop:
            logging.info(f'Early stopping with best loss: {earlyStop.bestLoss: .3f}')
            break
    plotTrnCurve(lossTrnLst, lossValLst, os.path.join(config['out'], 'training_curve.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configFile', type=Union[str, None], default=None,
                        help='the path of configure file determining the hyper-parameters')
    args = parser.parse_args()

    # prepare configures
    if args.configFile is not None:
        with open(args.configFile, 'r') as f:
            config = json.load(f)
    else:
        config = defaultdict()
        # arguments for dataloader
        config['data'] = {'xDataFile': 'data/data.pkl',   # input file containing forcing data and static attributes
                          'flowFile': 'data/streamflow.csv',  # streamflow file
                          'laiFile': 'data/LAI.csv',  # LAI file
                          # Training, validation, and testing periods
                          'periods': [['1981-1-1', '1994-12-31'], ['1995-1-1', '2004-12-31'],
                                      ['2005-1-1', '2019-12-31']],
                          # 'periods': [['1960-1-1', '1994-12-31'], ['1995-1-1', '2004-12-31'],
                          #             ['2005-1-1', '2019-12-31']],
                          'hydStations': ['MAQ'],  # which hydrological stations are used for training
                          'excludeBsnAttrVar': 'slope_min',  # exclude some static attributes
                          'spinUp': [365, 365, 365],  # spin-up period for training, validation, and testing
                          # sequence length, window size, mainstream index and padding values in the routing order array
                          'seqLen': 1460, 'winSz': 1460, 'mainIdx': 0, 'pad': 9999,
                          # sub-basin id for each hydrological station
                          'hydStationBsnIdx': {'HHY': 13, 'JIM': 45, 'MET': 53, 'MAQ': 35, 'JUG': 23, 'TNH': 0}}

        # hyperparameters for training
        if isinstance(config['data']['hydStations'], str):
            config['data']['hydStations'] = [config['data']['hydStations']]
        outBsnIdx = [config['data']['hydStationBsnIdx'][bsn] for bsn in config['data']['hydStations']]
        # get five random seeds using (np.random.uniform(low=0, high=1, size=6) * (10**6)).astype(int)
        # [668823, 759826, 211765, 908331, 808530, 178716]
        # cal_gs_LAI means whether to calculate LAI loss based on the growing season
        # snow_NN means whether to use NNs to learn the parameter of snow sublimation
        # nMul means number of multi-components in Feng et al., 2022,
        # wMul means whether the weight of the multi-components is automatically learnt by NNs
        config['train'] = {'logLoss': False, 'wLog': 0.25, 'wStationLoss': {'TNH': 1, 'MAQ': 1, 'JIM': 1, 'HHY': 0.5},
                           'wLAI': 0.5, 'patience': 20, 'lr': 0.005, 'clip': 3, 'epochs': 200, 'gpu': True,
                           'seed': 759826, 'outBsnIdx': outBsnIdx, 'dropout': 0.5, 'activFn': 'sigmoid',
                           'cal_gs_LAI': False, 'snow_NN': False, 'model_type': 'EXP-BGM', 'nMul': 16, 'wMul': True}
        # hyper-parameters for model
        config['staNet'] = {'type': 'ConvMLP',  # ['MLP', 'LSTM', 'LSTMMLP', 'ConvMLP']
                            'inFC': 134, 'hidFC': 128, 'outFC': 9, 'inLSTM': 5, 'hidLSTM': 128, 'outLSTM': 64,
                            'nAttr': 70, 'nMet': 5, 'lenMet': config['data']['spinUp'][0],
                            'nKernel': [10, 5, 1], 'kernelSz': [7, 5, 3], 'stride': [1, 1, 1], 'poolSz': [3, 2, 1]}
        config['dynNet'] = {'type': 'LSTM',  # ['LSTM', 'LSTMCell']
                            'inLSTM': 75, 'hidLSTM': 128, 'outLSTM': 5}
        config['routNet'] = {'inFC': 6, 'hidFC': 32, 'outFC': 2}

    # Configure output file path
    now = time.strftime('%m%d-%H%M', time.localtime())
    staType = config['staNet']['type']
    if staType == 'MLP':
        staSz = [config['staNet']['inFC'], config['staNet']['hidFC'], config['staNet']['outFC']]
    elif staType == 'LSTM':
        staSz = [config['staNet']['inLSTM'], config['staNet']['hidLSTM'], config['staNet']['outLSTM']]
    elif staType == 'LSTMMLP':
        staSz = [config['staNet']['inFC'], config['staNet']['hidFC'], config['staNet']['outFC'],
                 config['staNet']['inLSTM'], config['staNet']['hidLSTM'], config['staNet']['outLSTM']]
    else:
        staSz = [config['staNet']['nAttr'], config['staNet']['nMet'], config['staNet']['hidFC'],
                 config['staNet']['outFC'], config['staNet']['lenMet'], config['staNet']['nKernel'],
                 config['staNet']['kernelSz'], config['staNet']['stride'], config['staNet']['poolSz']]
    dynType = config['dynNet']['type']
    dynSz = [config['dynNet']['inLSTM'], config['dynNet']['hidLSTM'], config['dynNet']['outLSTM']]
    routSz = [config['routNet']['inFC'], config['routNet']['hidFC'], config['routNet']['outFC']]
    seed = config['train']['seed']
    hydStation = '-'.join([s[0] for s in config['data']['hydStations']])
    snow_NN = config['train']['snow_NN']
    nMul, wMul = config['train']['nMul'], config['train']['wMul']
    model_type = config['train']['model_type']
    config['out'] = f"./checkpoints/seed_{seed}_nMul_{nMul}_wMul_{wMul}_out_{hydStation}_wLAI_" \
                    f"{config['train']['wLAI']}_snow_NN_{snow_NN}_model_type_{model_type}_t_{now}"
    Path(config['out']).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config['out'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # fix the random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # configure log file
    logFile = os.path.join(config['out'], 'log.txt')
    if os.path.exists(logFile):
        os.remove(logFile)
    logging.basicConfig(filename=logFile, level=logging.INFO, format='%(asctime)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # determine the device
    device = torch.device('cuda:0' if config['train']['gpu'] and torch.cuda.is_available() else 'cpu')
    logging.info(f'{device} is used in training.')
    logging.info(f'The output path is {config["out"]}')

    # get dataloaders
    loader = DataLoader(xdataFile=config['data']['xDataFile'], flowFile=config['data']['flowFile'],
                        laiFile=config['data']['laiFile'], periods=config['data']['periods'],
                        spinUp=config['data']['spinUp'], seqLen=config['data']['seqLen'],
                        winSize=config['data']['winSz'], device=device,
                        hydStation=config['data']['hydStations'], excludeBsnAttrVar=config['data']['excludeBsnAttrVar'])

    # configure model
    model = Net(staType=staType, staSz=staSz, dynType=dynType, dynSz=dynSz, routSz=routSz, model_type=model_type,
                routOrder=loader.routOrder, area=loader.area, upIdx=loader.dataAll['rivUpIdx'], nMul=nMul, wMul=wMul,
                mainIdx=config['data']['mainIdx'], pad=config['data']['pad'], dropout=config['train']['dropout'],
                activFn=config['train']['activFn'], device=device, snow_NN=snow_NN)

    model = model.to(device)
    wStation = [config['train']['wStationLoss'][station] for station in config['data']['hydStations']]
    lossFn = nseLoss(obsLAI=loader.LAI, logLossOpt=config['train']['logLoss'], wLog=config['train']['wLog'],
                     wStation=wStation, wLAI=config['train']['wLAI'], cal_gs_LAI=config['train']['cal_gs_LAI'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    earlyStop = earlyStopping(savePath=os.path.join(config['out'], 'model.pt'), patience=config['train']['patience'],
                              delta=0.0002)
    trainModel(model=model, loaderTrn=loader.loaderTrn, loaderVal=loader.loaderVal, lossFn=lossFn, optimizer=optimizer,
               scheduler=scheduler, earlyStop=earlyStop, config=config)

    # get and save simulations
    model.load_state_dict(torch.load(os.path.join(config['out'], 'model.pt')))
    saveSimulation(model=model, loaderVal=loader.loaderVal, loaderTst=loader.loaderTst, outPath='simulation.pkl',
                   config=config, obsLAI=loader.LAI, cal_gs_LAI=config['train']['cal_gs_LAI'], model_type=model_type)

    # os._exit(0)
