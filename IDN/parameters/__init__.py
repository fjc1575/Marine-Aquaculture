import argparse
from parameters import basic_training_params
from parameters import wandb_parmas


def setupArgs():

    parser = argparse.ArgumentParser()
    parser = basic_training_params.addparameters(parser)
    parser = wandb_parmas.addparameters(parser)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    a = setupArgs()

    print(a.cam_method)