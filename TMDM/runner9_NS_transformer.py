import argparse
import torch
from model9_NS_transformer.exp.exp_main import Exp_Main
import random
import numpy as np
import setproctitle


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    setproctitle.setproctitle('ILI_thread')

    parser = argparse.ArgumentParser(
        description='Residual TMDM for Time Series Forecasting'
    )

    # basic config
    parser.add_argument('--is_training', type=str2bool, default=True, help='status')
    parser.add_argument('--model_id', type=str, default='ETTh2_96_192', help='model id')
    parser.add_argument('--model', type=str, default='TMDM-r', help='model name')
    parser.add_argument(
        '--base_model',
        type=str,
        default='ns_Transformer',
        choices=['ns_Transformer', 'ns_Autoformer'],
        help='base predictor for y_base'
    )

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument(
        '--features', type=str, default='M',
        help='forecasting task, options:[M, S, MS]; '
             'M:multivariate predict multivariate, '
             'S:univariate predict univariate, '
             'MS:multivariate predict univariate'
    )
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument(
        '--freq', type=str, default='h',
        help='freq for time features encoding, options:[s,t,h,d,b,w,m], '
             'or detailed freq like 15min or 3h'
    )
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument(
        '--distil', action='store_false',
        help='whether to use distilling in encoder',
        default=True
    )
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument(
        '--embed', type=str, default='timeF',
        help='time features encoding, options:[timeF, fixed, learned]'
    )
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--k_z', type=float, default=1e-2, help='KL weight')
    parser.add_argument('--k_cond', type=float, default=1, help='Condition weight')
    parser.add_argument('--d_z', type=int, default=8, help='latent dim')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of test input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer lr for diffusion model')
    parser.add_argument('--learning_rate_Cond', type=float, default=0.0001, help='optimizer lr for base model')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_residual_tmdm', type=int, default=1)
    parser.add_argument('--lambda_prior', type=float, default=0.1)
    parser.add_argument('--freeze_base_model', type=str2bool, default=True, help='whether to freeze y_base model')

    # new: residual mean consistency loss weight
    parser.add_argument('--lambda_rmc', type=float, default=0.05,help='weight of residual mean consistency loss')
    #
    parser.add_argument('--lambda_r0', type=float, default=0.1,
                    help='weight for residual r0 reconstruction loss')

    parser.add_argument('--simpatch_d_model', type=int, default=128,
                        help='hidden dimension for residual patch denoiser')

    parser.add_argument('--simpatch_layers', type=int, default=1,
                        help='number of Transformer encoder layers in residual patch denoiser')

    parser.add_argument('--simpatch_heads', type=int, default=4,
                        help='number of attention heads in residual patch denoiser')

    parser.add_argument('--simpatch_d_ff', type=int, default=256,
                        help='feed-forward dimension in residual patch denoiser')

    parser.add_argument('--patch_len', type=int, default=16,
                        help='patch length for residual patch denoiser')

    parser.add_argument('--stride', type=int, default=8,
                        help='patch stride for residual patch denoiser')
    parser.add_argument("--sample_temperature",type=float,default=1.0,help="temperature for residual diffusion sampling noise",)
    #MOM
        # Sample aggregation for point forecasting
    parser.add_argument(
        "--point_agg",
        type=str,
        default="mean",
        choices=["mean", "median", "trimmed_mean", "mom"],
        help="aggregation method over diffusion samples for point forecasting metrics",
    )

    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="trim ratio for trimmed_mean aggregation",
    )

    parser.add_argument(
        "--mom_groups",
        type=int,
        default=5,
        help="number of groups for Median-of-Means aggregation",
    )

    parser.add_argument(
        "--mom_repeats",
        type=int,
        default=3,
        help="number of random repetitions for Median-of-Means aggregation",
    )
    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # diffusion related args
    parser.add_argument('--diffusion_config_dir', type=str, default='./model9_NS_transformer/configs/toy_8gauss.yml')
    parser.add_argument('--cond_pred_model_pertrain_dir', type=str,
                        default='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth')
    parser.add_argument('--CART_input_x_embed_dim', type=int, default=32, help='feature dim for x in diffusion model')
    parser.add_argument('--mse_timestep', type=int, default=0)
    parser.add_argument('--MLP_diffusion_net', type=str2bool, default=False, help='use MLP or Unet')
    parser.add_argument('--timesteps', type=int, default=1000)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    fix_seed = np.random.randint(2147483647) if args.seed == -1 else args.seed
    print('Using seed:', fix_seed)

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_base{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_lamrmc{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.base_model,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.lambda_rmc,
                args.des,
                ii
            )

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_base{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_lamrmc{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.base_model,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.lambda_rmc,
            args.des,
            ii
        )

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
