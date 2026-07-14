import os


def addparameters(parser):
    #网络结构相关参数segnetdeeplabv3plus
    parser.add_argument('--arch', type=str,default = 'fcn8s', help='网络结构:segnet,unet,pspnet,deeplabv3plus,cmtfnet,maresunet,mdoaunet,unetplusplus,fcn8s,hrnet,swinupernet,segformer,swinunet,pcamnet')

    parser.add_argument('--in_ch', type=int, default=3, help='input image channels')

    parser.add_argument('--bin', type=int, default=4, help='bin')

    parser.add_argument('--gamma', type=int, default=1, help='gamma')

    parser.add_argument('--num_class', type=int, default=2, help='output image channels')
    #数据集
    parser.add_argument('--dataset', type=str, default='GF3', help='GF3,RADARSAT-2')

    parser.add_argument('--augmentation', type=bool, default=True, help='是否进行数据增强')

    parser.add_argument('--lr',                default=0.0003,  type=float,        help='网络参数的学习率。')

    parser.add_argument('--n_epochs',          default=150,      type=int,          help='训练周期数。')

    parser.add_argument('--bs',                default=8,     type=int,          help='Mini-Batchsize大小。')

    parser.add_argument('--seed',              default=0,        type=int,          help='随机种子,用于重现结果。')

    parser.add_argument('--gpu',          default=[0], nargs='+',                  type=int, help='要使用的GPU。')

    parser.add_argument('--data_source_path', default=r'C:\polsar\datasets\dataset', type=str, help='训练资源存放的位置')
    parser.add_argument('--model_save_path', default=r'C:\polsar\resource\MPA_RUet_Source\model', type=str, help='训练模型的保存位置。')
    parser.add_argument('--train_result_path', default=r'C:\polsar\resource\MPA_RUet_Source\train_result', type=str, help='训练图片存储位置')
    parser.add_argument('--test_result_path', default=r'C:\polsar\resource\MPA_RUet_Source\test_result', type=str, help='测试图片存储位置')


    return parser
