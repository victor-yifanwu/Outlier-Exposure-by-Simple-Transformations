import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from openood.datasets import get_dataloader
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
import openood.utils.comm as comm
from openood.utils import Config
from oest.cifar10.utils import *

def train(net, args, logger=None):
    config_files = [
        './configs/datasets/cifar10/cifar10.yml',
        './configs/datasets/cifar10/cifar10_ood.yml',
        './configs/networks/resnet18_32x32.yml',
        './configs/pipelines/test/test_ood.yml',
        './configs/preprocessors/oest_preprocessor.yml',
        './configs/postprocessors/ebo.yml',
    ]
    config = Config(*config_files)
    config.dataset.train.batch_size = args.batch_size
    config.parse_refs()
    
    id_loader = get_dataloader(config)
    pd_loader = get_dataloader(config)
    
    net.train()
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.max_epoch * len(id_loader['train']),
            1,
            1e-6,
        ),
    )
    
    for epoch_idx in range(1, args.max_epoch + 1):
        net.train()
        id_dataiter = iter(id_loader['train'])
        pd_dataiter = iter(pd_loader['train'])

        progress_bar = tqdm(
            range(1, len(id_dataiter) + 1),
            desc='Epoch {:02d}: '.format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
            ascii=True
        )
        
        for _ in progress_bar:
            # id data prepare
            id_batch = next(id_dataiter)
            id_data = id_batch['data'].cuda()
            id_target = id_batch['label'].cuda()
            
            # pd data prepare
            pd_batch = next(pd_dataiter)
            pd_data = pd_batch['data'].cuda()
            data_blur = transform.blur(pd_data)
            data_cutout = transform.cutout(pd_data)
            data_noise = transform.noise(pd_data)
            data_perm = transform.perm(pd_data)
            data_rotate = transform.rotate(pd_data)
            data_sobel = transform.sobel(pd_data)
            pd_data = torch.cat((data_cutout, data_sobel, data_noise, data_blur, data_perm, data_rotate), dim=0)
            pd_data = pd_data[torch.randperm(pd_data.size(0))]
            pd_data = pd_data[:len(id_data)]
            
            data = torch.cat((id_data, pd_data), dim=0)

            # forward
            output = net(data)
            loss = F.cross_entropy(output[:len(id_data)], id_target)
            Ec_out = - torch.logsumexp(output[len(id_data):], dim=1)
            Ec_in = - torch.logsumexp(output[:len(id_data)], dim=1)
            loss += - args.alpha * torch.log(torch.sigmoid((Ec_out.mean() - Ec_in.mean()) / args.beta))
                    
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # print lr
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({'lr': current_lr})
            progress_bar.update(1)
            
    torch.save(net.state_dict(), os.path.join(args.model_folder, f'checkpoint_final.pth'))
    eval(net, logger=logger)
    

def eval(net, logger=None):
    net.eval()
    evaluator = Evaluator(
        net,
        id_name='cifar10',                     # the target ID dataset
        data_root='./data',                    # change if necessary
        config_root=None,                      # see notes above
        preprocessor=None,                     # default preprocessing for the target ID dataset
        postprocessor_name="ebo",              # the postprocessor to use
        postprocessor=None,                    # if you want to use your own postprocessor
        batch_size=200,                        # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=2)                         # could use more num_workers outside colab
    metrics = evaluator.eval_ood(fsood=False)
    if logger is not None:
            logger.info(metrics)
            
def main(args):
    set_seed(args.seed)
    save_folder = '%s_%s' % (args.dataset, args.affix)
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')
    print_args(args, logger)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ResNet18_32x32(num_classes=10).to(device)
    net.load_state_dict(
        torch.load(args.load_checkpoint)
    )

    net.cuda()
    net.train()

    if args.todo == 'train':
        train(net, args, logger)
    elif args.todo == 'test':
        eval(net, logger)

            
if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)