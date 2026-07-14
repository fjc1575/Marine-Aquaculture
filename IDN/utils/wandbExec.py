import wandb


def wandb_init(opt):
    if not wandb.run:
        wandb.login(key=opt.wandb_key)
    opt.name = opt.arch + '_' + opt.dataset
    wandb.init(project=opt.project,
            config={
                   "learning_rate": opt.lr,
                   "batch_size": opt.bs,
                   "epochs": opt.n_epochs,
                   "architecture": opt.arch,
               },
            name=opt.name,
            group=opt.group,


            )

    # wandb.config.update(opt)
def wandb_log(res,step):
    #数据展示
    wandb.log(res, step=step)
    #图像展示

