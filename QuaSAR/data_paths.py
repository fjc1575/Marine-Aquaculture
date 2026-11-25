"""Apply a user-supplied data root to the bundled dataset layouts."""
import os


def apply_data_root(config, data_root, task):
    root = os.fspath(data_root).replace('\\', '/')
    if task not in ('segmentation', 'detection'):
        raise ValueError(f'Unknown task: {task}')

    def update(dataset, split):
        if isinstance(dataset, (list, tuple)):
            for child in dataset:
                update(child, split)
        elif 'dataset' in dataset:
            update(dataset['dataset'], split)
        elif 'datasets' in dataset:
            update(dataset['datasets'], split)
        elif task == 'segmentation':
            dataset['data_root'] = root
        else:
            suffix = 'train2017' if split == 'train' else 'val2017'
            dataset['ann_file'] = os.path.join(root, 'annotations', f'instances_{suffix}.json').replace('\\', '/')
            dataset['img_prefix'] = os.path.join(root, suffix).replace('\\', '/') + '/'
    config.data_root = root
    for split in ('train', 'val', 'test'):
        if split in config.data:
            update(config.data[split], split)
