import os
import numpy as np
import torch
import torch.utils.data as data
from torch.nn import functional as F
from datetime import datetime

from learners.pmo_utils import Pool


class CFSTDataset(data.Dataset):

    def __init__(self, root,
                 train=True, transform=None,
                 download_flag=False, lab=True, swap_dset = None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5,
                 mode='continual', return_concepts=False, first_split_size=1, other_split_size=1,
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
        self.return_concepts = return_concepts
        self.first_split_size = first_split_size
        self.other_split_size = other_split_size

        # load dataset
        self.benchmark = None
        self.download_flag = download_flag
        self.load()
        self.num_classes = self.benchmark.n_classes
        if self.train:
            self.target_datasets = self.benchmark.train_datasets    # train_datasets; debug to use val_datasets
        elif self.validation:
            self.target_datasets = self.benchmark.val_datasets
        else:
            self.target_datasets = self.benchmark.test_datasets

        # for concepts
        self.target_sample_info = None
        if len(self.benchmark.label_info) == 4 and return_concepts:
            if 'concept_set' in self.benchmark.label_info[3].keys():
                self.num_concepts = len(self.benchmark.label_info[3]['concept_set'])
                print(f'num_concepts: {self.num_concepts}.')
                self.map_int_label_to_concept = self.benchmark.label_info[3]['map_int_label_to_concept']
                if 'train_list' in self.benchmark.label_info[3].keys():      # continual
                    if self.train:
                        self.target_sample_info = self.benchmark.label_info[3]['train_list']
                    elif self.validation:
                        self.target_sample_info = self.benchmark.label_info[3]['val_list']
                    else:
                        self.target_sample_info = self.benchmark.label_info[3]['test_list']
                else:       # CFST
                    self.target_sample_info = self.benchmark.label_info[3]['img_list']

        # Pool
        self.memory = Pool(0, self.seed)   # memory size will change in update_coreset()

        # define task
        self.dataset = None
        self.t = 0     # task id

    def debug_mode(self):
        self.target_datasets = self.benchmark.val_datasets
        if len(self.benchmark.label_info) == 4 and self.return_concepts:
            if 'train_list' in self.benchmark.label_info[3].keys():
                self.target_sample_info = self.benchmark.label_info[3]['val_list']

    def __getitem__(self, index, simple = False):
        data = self.dataset[index]
        if len(data) == 3:
            img, target, ori_idx = data
            t = self.t
        else:       # img, target, ori_idx, task
            img, target, ori_idx, t = data

        # concept label
        if self.target_sample_info is not None:
            concepts, _, _ = self.get_concepts(ori_idx)
            return img, target, concepts, t

        return img, target, t

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return 'Dataset ' + self.__class__.__name__

    def load_dataset(self, t, train=True, ignore_split=False):
        """set specific task
        train=True -> only load task t; False -> load task <=t.
        NOTE: train=False will be a bug if in task-IL.
        Originally, CFST dataset has 10/3 10-way tasks.
        The load task t is according to first_split_size and other_split_size.
        If ignore_split, dataset will be based on origin CFST dataset, and ignore train flag.
        """
        if ignore_split:
            self.dataset = self.target_datasets[t]
        else:
            # check load task range
            if t == 0:
                s = 0
                f = self.first_split_size
            elif train:
                s = self.first_split_size + (t - 1) * self.other_split_size
                f = self.first_split_size + t * self.other_split_size
            else:
                s = 0
                f = self.first_split_size + t * self.other_split_size

            load_task_range = [ti for ti in range(s, f)]

            if len(load_task_range) == 1:
                self.dataset = self.target_datasets[load_task_range[0]]
            else:
                self.dataset = torch.utils.data.ConcatDataset(
                    [self.target_datasets[ti] for ti in load_task_range])
                self.dataset.targets = np.concatenate([self.target_datasets[ti].targets for ti in load_task_range])
        self.t = t

    def get_single_class_dataset(self, label):
        """from dataset load images with given label"""
        if hasattr(self.dataset, 'targets'):
            targets = np.array(self.dataset.targets)
        else:
            targets = np.concatenate([self.dataset.datasets[t].targets for t in self.dataset.datasets])
        cls_indices = np.where(targets == label)[0]
        return torch.utils.data.Subset(self, cls_indices)

    def get_unique_labels(self):
        targets = self.dataset.targets
        return np.unique(targets)

    def update_pool(self, pool_size, task_id=None, pool=None):
        if task_id is None:
            task_id = self.t
        if pool is None:
            pool = self.memory
        print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Update pool for task {task_id}.')

        # update pool.memory_size
        pool.memory_size = pool_size

        # put images to pool
        if len(self.target_datasets[task_id].images) > 0:    # already loaded
            images = self.target_datasets[task_id].images
            targets = self.target_datasets[task_id].targets
            tasks = np.array([task_id for _ in range(len(images))])
        else:       # load one by one
            images, targets, tasks = [], [], []
            for item, (img, target) in enumerate(self.target_datasets[task_id]):
                # img: tensor, target: numpy, task: numpy or int
                images.append(img)
                targets.append(target)
                tasks.append(task_id)
            images = torch.stack(images)
            targets = np.array(targets)
            tasks = np.array(tasks)

        pool.put(images, {'labels': targets, 'tasks': tasks})

        print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] DONE.')

    def append_coreset(self, only=False, interp=False):
        if self.train and (len(self.memory) > 0):
            if only:
                self.dataset = self.memory
            else:
                coreset = self.memory.return_random_dataset(size=int(len(self.dataset)))
                # coreset = self.memory
                self.dataset = torch.utils.data.ConcatDataset([self.dataset, coreset])

    def update_coreset(self, coreset_size, seen):
        print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Update coreset for task {self.t}.')
        if self.mode != 'continual':    # for few-shot testing, do not use replay
            print(f'Not continual mode (current {self.mode}), pass')
            return

        self.update_pool(coreset_size, self.t, self.memory)

    def relabel2str(self, relabel):
        """Valid only on class-IL setting"""
        original_classes_in_exp = self.benchmark.original_classes_in_exp    # [[real_label]]
        classes_in_exp = self.benchmark.classes_in_exp      # [[0-9], [10-19],...]
        label_info = self.benchmark.label_info
        ori_label_2_str = label_info[2]

        # find relabel to ori label
        task_id = None
        label_id = None
        for _task_id, task in enumerate(classes_in_exp):
            if relabel in task:
                task_id = _task_id
                label_id = list(task).index(relabel)
                break
        if task_id is None:
            return None

        ori_label = original_classes_in_exp[task_id][label_id]
        label_str = ori_label_2_str[ori_label]
        return label_str, [task_id, label_id]

    def get_concepts(self, index=-1, mode='label'):
        """Obtain concepts [Option:
        'label': concept labels [1, 2] # additional dim for dataloader;
        'mask': img mask [224, 224] ]
        """
        if self.target_sample_info is not None and mode in ['label', 'mask']:
            map_int_concepts_label_to_str = self.benchmark.label_info[3]['map_int_concepts_label_to_str']
            map_int_label_to_concept = self.benchmark.label_info[3]['map_int_label_to_concept']     # exact label
            original_classes_in_exp = self.benchmark.original_classes_in_exp.flatten()
            if index >= 0:
                concepts = self.target_sample_info[index][2]    # e.g., [10, 15]
                concepts_str = [map_int_concepts_label_to_str[idxx] for idxx in concepts]
                if mode == 'label':
                    concepts = torch.tensor(concepts).long()
                    concepts = self.process_concepts(concepts, self.num_concepts)
                    concepts = concepts.reshape(1, concepts.size(0))    # shape [1, 21]
                    return concepts, None, concepts_str
                elif mode == 'mask':
                    img_shape = self.benchmark.x_dim[1:]        # [3, 224, 224] -> [224, 224]
                    concepts = torch.tensor(concepts).long()
                    position = self.target_sample_info[index][3]    # e.g., [4, 3]
                    position = torch.tensor(position).long()
                    mask = torch.zeros(4).long()
                    mask[position-1] = concepts
                    mask[mask == 0] = self.num_concepts
                    mask = mask.reshape(2, 2).repeat_interleave(
                        img_shape[0]//2, dim=0).repeat_interleave(img_shape[1]//2, dim=1)
                    return mask, position.numpy().tolist(), concepts_str
            else:   # return all label's concept
                concepts = [
                    map_int_label_to_concept[original_classes_in_exp[label]]
                    for label in range(len(map_int_label_to_concept.keys()))]
                # [n_cls * [list of concepts: e.g., 1, 10]]
                
                # concepts_list = [
                #     self.process_concepts(
                #         torch.tensor(self.target_sample_info[idx][2]).long(), self.num_concepts
                #     ) for idx in range(len(self))]
                # concepts_list = torch.stack(concepts_list)  # [len_datasets, n_concepts]

                return concepts

        return None, None, None

    def process_concepts(self, concepts, num_prompts):
        # e.g, from [0, 1, 2] -> [num_prompts]  multi-hot float
        concept_labels = F.one_hot(concepts, num_prompts)   # [n_c, num_prompts]
        concept_labels = torch.sum(concept_labels, dim=0).float()   # [num_prompts]

        return concept_labels

    def load(self):
        """need to implement,
        load benchmark contain train_datasets, test_datasets, val_datasts"""
        pass


class CGQA(CFSTDataset):

    def load(self):
        from dataloaders import cgqa
        load_set = None
        if self.download_flag:
            load_set = 'train' if self.train else ('val' if self.validation else 'test')
        if self.mode == 'continual':
            self.benchmark = cgqa.continual_training_benchmark(
                10, image_size=(224, 224), return_task_id=False,
                seed=self.seed,
                train_transform=cgqa._build_default_transform(image_size=(224, 224), is_train=True),
                eval_transform=cgqa.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                memory_size=0,
                load_set=load_set,
            )
        else:
            self.benchmark = cgqa.fewshot_testing_benchmark(
                100, image_size=(224, 224), mode=self.mode, task_offset=10,
                seed=self.seed,
                train_transform=cgqa.build_transform_for_vit(is_train=False),
                eval_transform=cgqa.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                load_set=load_set,
            )


class COBJ(CFSTDataset):

    def load(self):
        from dataloaders import cobj
        load_set = None
        if self.download_flag:
            load_set = 'train' if self.train else ('val' if self.validation else 'test')
        if self.mode == 'continual':
            self.benchmark = cobj.continual_training_benchmark(
                3, image_size=(224, 224), return_task_id=False,
                seed=self.seed,
                train_transform=cobj._build_default_transform(image_size=(224, 224), is_train=True),
                eval_transform=cobj.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                memory_size=0,
                load_set=load_set,
            )
        else:
            self.benchmark = cobj.fewshot_testing_benchmark(
                100, image_size=(224, 224), mode=self.mode, task_offset=3,
                seed=self.seed,
                train_transform=cobj.build_transform_for_vit(is_train=False),
                eval_transform=cobj.build_transform_for_vit(is_train=False),
                dataset_root=os.path.join(self.root, 'CFST'),
                load_set=load_set,
            )
