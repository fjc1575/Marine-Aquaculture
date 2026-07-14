

def addparameters(parser):
    ### 在线记录/Wandb日志参数
    parser.add_argument('--log_online', type = bool,default = False,help='如果设置,除了离线日志外,运行指标也会在线存储。通常应该设置。')
    parser.add_argument('--wandb_key', default='dba128fa15ca78309798e5b1ba7036d7a096b52a', type=str,help='W&B的API密钥。')
    parser.add_argument('--project', default='Ablation', type=str,help='gf_shandong,radarsat2,Ablation')
    parser.add_argument('--name', default='Ablation', type=str, help='实验名称，用于标识一个特定的实验。')
    parser.add_argument('--group', default='Ablation', type=str, help='将多个实验分组在一起，方便比较和分析。')
    return parser
