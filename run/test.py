import logging
import torch
from torch.utils.data import DataLoader
from models.EGIInet import EGIInet

#import utils.data_loaders
#import utils.helpers
from tqdm import tqdm

from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader

        ViPC_test = ViPCDataLoader(r'/project/EGIInet/test_list.txt',
                                   data_path=cfg.DATASETS.SHAPENET.VIPC_PATH,
                                   status='test',
                                   view_align=False, category=cfg.TEST.CATE)
        test_data_loader = DataLoader(ViPC_test,
                                     batch_size=cfg.TEST.BATCH_SIZE,
                                     num_workers=cfg.CONST.NUM_WORKERS,
                                     shuffle=True,
                                     drop_last=True,
                                     prefetch_factor=cfg.CONST.DATA_perfetch)

    # Setup networks and initialize networks
    if model is None:
        model = EGIInet()#.apply(weights_init_normal)
        model.cuda()
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])
    test_metrics = AverageMeter(['CDl1', 'CDl2', 'F1'])
    category_metrics = dict()

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (view,gt_pc,part_pc) in enumerate(t):

            with torch.no_grad():

                partial = part_pc.cuda()  # [16,2048,3]
                gt = gt_pc.cuda()  # [16,2048,3]
                png = view.cuda()
                partial = farthest_point_sample(partial,cfg.DATASETS.SHAPENET.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.SHAPENET.N_POINTS)

                model.eval()
              
                pcds_pred,_ = model(partial, png)
                cdl1, cdl2, f1 = calc_cd(pcds_pred, gt, calc_f1=True)
                   
                cdl1 = cdl1.mean().item() * 1e3
                cdl2 = cdl2.mean().item() * 1e3
                f1 = f1.mean().item()

                _metrics = [cdl1, cdl2, f1]
                test_losses.update([cdl1, cdl2, f1])

                test_metrics.update(_metrics)

    # Print testing results
    print('============================ TEST RESULTS ============================')

    """
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    """
    
    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)