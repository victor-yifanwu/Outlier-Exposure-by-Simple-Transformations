import argparse

def parser():
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'test'], default='finetune',
        help='what behavior want to do: finetune | test')
    parser.add_argument('--dataset', default='cifar-100', help='use what dataset')
    parser.add_argument('--data_root', default='/home/yilin/Data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='./oest/cifar100/log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='./oest/cifar100/checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./cifar100_resnet18_32x32/s0/last_epoch100.ckpt')
    parser.add_argument('--affix', default='default', help='the affix for the save folder')

    parser.add_argument('--seed', '-s', type=int, default=1)
    
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=10, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
        help='the parameter of l2 restriction for weights')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--n_eval_step', type=int, default=100, 
        help='number of iteration per one evaluation')
    
    parser.add_argument('--alpha', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=0.1)
    
    parser.add_argument('--ablation', default=None)
    
    parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))