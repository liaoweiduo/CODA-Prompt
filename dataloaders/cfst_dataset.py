import os
import numpy as np
import torch
import torch.utils.data as data

from learners.pmo_utils import Pool


class CFSTDataset(data.Dataset):

    def __init__(self, root,
                 train=True, transform=None,
                 download_flag=False, lab=True, swap_dset = None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5,
                 mode='continual',
                 ):

        # process rest of args
        self.root = os.path.expanduser(root)
        # self.transform = transform      # no use
        self.train = train  # training set or test set
        self.validation = validation        # if val, load val set instead of test set
        self.seed = seed
        # self.tasks = tasks      # load specific true_classes; no use
        # self.download_flag = download_flag  # no use
        self.mode = mode        # [continual, sys, pro, sub, non, noc]

        # load dataset
        self.load()
        self.num_classes = self.benchmark.n_classes

        # Pool
        self.pool = Pool(0, self.seed)   # memory size will change in update_coreset()

        # define task
        self.dataset = None
        self.t = -1     # task id

    def __getitem__(self, index, simple = False):
        img, target = self.dataset[index]
        return img, target, self.t

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return 'Dataset ' + self.__class__.__name__

    def load_dataset(self, t, train=True):
        """set specific task
        train=True -> only load task t; False -> load task <=t.
        NOTE: train=False will be bug if in task-IL.
        """
        if self.train:
            target = self.benchmark.train_datasets
        elif self.validation:
            target = self.benchmark.val_datasets
        else:
            target = self.benchmark.test_datasets

        if train:
            self.dataset = target[t]
        else:
            self.dataset = torch.utils.data.ConcatDataset([target[s] for s in range(t+1)])
        self.t = t

    def append_coreset(self, only=False, interp=False):
        if self.train and (len(self.pool) > 0):
            if only:
                self.dataset = self.pool
            else:
                coreset = self.pool.return_random_dataset(size=len(self.dataset))
                self.dataset = torch.utils.data.ConcatDataset([self.dataset, coreset])

    def update_coreset(self, coreset_size, seen):
        print(f'Update coreset for task {self.t}.')
        if self.mode != 'continual':    # for few-shot testing, do not use replay
            print(f'Not continual mode (current {self.mode}), pass')
            return

        # update pool.memory_size
        self.pool.memory_size = coreset_size

        # put images to pool
        images, targets, tasks = [], [], []
        for item, (img, target) in enumerate(self.dataset):
            # img: tensor, target: numpy, task: numpy or int
            images.append(img)
            targets.append(target)
            tasks.append(self.dataset.get_task_label(item))
        images = torch.stack(images)
        targets = np.array(targets)
        tasks = np.array(tasks)

        self.pool.put(images, {'labels': targets, 'tasks': tasks})

    def load(self):
        """need to implement,
        load benchmark contain train_datasets, test_datasets, val_datasts"""
        pass


class CGQA(CFSTDataset):

    def load(self):
        from dataloaders import cgqa
        if self.mode == 'continual':
            self.benchmark = cgqa.continual_training_benchmark(
                10, image_size=(224, 224), return_task_id=False,
                seed=self.seed,
                train_transform=cgqa.build_transform_for_vit(is_train=True),
                eval_transform=cgqa.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                memory_size=0,
                load_set='train' if self.train else ('val' if self.validation else 'test'),
            )
        else:
            self.benchmark = cgqa.fewshot_testing_benchmark(
                300, image_size=(224, 224), mode=self.mode, task_offset=10,
                seed=self.seed,
                train_transform=cgqa.build_transform_for_vit(is_train=True),
                eval_transform=cgqa.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
            )


class COBJ(CFSTDataset):

    def load(self):
        from dataloaders import cobj
        if self.mode == 'continual':
            self.benchmark = cobj.continual_training_benchmark(
                3, image_size=(224, 224), return_task_id=False,
                seed=self.seed,
                train_transform=cobj.build_transform_for_vit(is_train=True),
                eval_transform=cobj.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                memory_size=0,
                load_set='train' if self.train else ('val' if self.validation else 'test'),
            )
        else:
            self.benchmark = cobj.fewshot_testing_benchmark(
                300, image_size=(224, 224), mode=self.mode, task_offset=3,
                seed=self.seed,
                train_transform=cobj.build_transform_for_vit(is_train=True),
                eval_transform=cobj.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
            )
