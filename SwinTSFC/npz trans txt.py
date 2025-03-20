import os


def write_name(np, tx):
    # npz文件路径
    files = os.listdir(np)
    print(files)
    # txt文件路径
    f = open(tx, 'w')
    for i in files:
        # name = i.split('\\')[-1]
        name = i[:-4] + '\n'
        f.write(name)


# write_name('./data/Synapse/train7_npz', './lists/lists_Synapse/train_7.txt')
write_name('./data/Synapse/changhaixian/trainl', './lists/lists_Synapse/changhaixian/train.txt')

