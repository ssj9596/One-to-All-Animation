from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional
import os
import numpy as np
from tqdm import tqdm
import tarfile
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from .aspect import get_num_pixels
from .bucket import Bucket, bucket_config, valid_bucket_config, valid_bucket_configs
def is_path_valid(path):
    if os.path.isfile(path):
        return True
    else:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if not dirname.endswith('.tar'):
            return False
        else:
            with tarfile.open(dirname) as tar:
                names = tar.getnames()
                if not basename in names:
                    return False
        return True

# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None, train_fps=16):
    if 'fps' in data and not np.isnan(data['fps']) and data['fps'] != train_fps:
        frame_interval = int(np.round(data['fps'] / train_fps))
        if frame_interval == 0:
            frame_interval = 1
    #TODO: 调整这里的最长帧数
    if 'start_frame' in data and 'end_frame' in data and not pd.isna(data['start_frame']) and not pd.isna(data['end_frame']):
        num_frames = data['end_frame'] - data['start_frame']
    else:
        num_frames = data['num_frames']
    
    if 'fps' in data and not np.isnan(data['fps']):
        num_frames = min(num_frames, int(10 * data['fps']))
    height = data['height']
    width = data['width']
    return method(
        num_frames,
        height,
        width,
        frame_interval,
        seed + data["id"] * num_bucket,
    )

def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

#NOTE: 这部分是否可以优化？现在使用的是Sampler，构建单进程读取数据的index list，然后依靠acceleretor.prepare迁移到多进程环境中。
# resume dataloader依赖的是accelerator.skip_first_batches。缺陷是这里不能实现每个epoch的shuffle，因为难以恢复epoch的内在随机种子，但是本身训练的epoches数量会很少，因此影响不大
class VariableVideoBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        world_size,
        seed: int = 42,
        drop_last: bool = False,
        verbose: bool = False,
        shuffle = True,
        num_bucket_build_workers: int = 1,
        train_fps = 16,
        valid = False,
        bucket_name = "default"
    ) -> None:
        self.dataset = dataset
        if not valid:
            self.bucket = Bucket(valid_bucket_configs[bucket_name])
        else:
            self.bucket = Bucket(valid_bucket_configs[bucket_name])
        self.verbose = verbose
        self.drop_last = drop_last
        self.seed = seed
        self.shuffle = shuffle
        self.num_bucket_build_workers = num_bucket_build_workers
        self.world_size = world_size 
        self.local_batch_index_list = None
        self.train_fps = train_fps

    def get_sample_index_list(self):
        bucket_sample_dict = self.group_by_bucket()
        if self.verbose:
            #TODO: 这里不准确，因为后面还有drop
            self._print_bucket_info(bucket_sample_dict)
        g = torch.Generator()
        #TODO: 如果可能，每个epoch更换随机种子
        g.manual_seed(self.seed)
        world_batch_index_list = []
        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            bs_world = self.world_size * bs_per_gpu
            if len(data_list) < bs_world:
                continue
            remainder = len(data_list) % bs_world
            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_world - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list
            # 每个bucket，按照world_bs进行分块，保证每个step，所有的gpu读取的都是同一个bucket的数据
            assert len(data_list) % bs_world == 0
            for idx in range(0, len(data_list), bs_world):
                world_batch_index_list.append(data_list[idx:idx+bs_world])
        if self.shuffle:
            batch_indices = torch.randperm(len(world_batch_index_list), generator=g).tolist()
            world_batch_index_list = [world_batch_index_list[i] for i in batch_indices]
        # 每个world batch，划分每张卡上的local batch
        local_batch_index_list = []
        for world_batch in world_batch_index_list:
            assert len(world_batch) % self.world_size == 0
            batch_size = len(world_batch) // self.world_size
            for idx in range(0, len(world_batch), batch_size):
                local_batch_index_list.append(world_batch[idx:idx+batch_size])
        return local_batch_index_list

    def __iter__(self) -> Iterator[List[int]]:
        if self.local_batch_index_list is None:
            self.local_batch_index_list = self.get_sample_index_list()
        local_batch_index_list = self.local_batch_index_list
        for batch in local_batch_index_list:
            real_t, real_h, real_w = self.real_thw[batch[0]]
            batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in batch]
            yield batch   
        self.local_batch_index_list = None #每个epoch之后重新开始
        self.seed += 1

    def __len__(self) -> int:
        #TODO: 优化一下
        # local_batch_index_list = self.get_sample_index_list()
        if self.local_batch_index_list is None:
            self.local_batch_index_list = self.get_sample_index_list()
        local_batch_index_list = self.local_batch_index_list
        return len(local_batch_index_list)

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()
        # bucket_id = self.bucket.get_bucket_id(100, 720, 1280)
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        print("Building buckets...")
        # parallel_apply == quickly apply
        # 3.7 use apply
        bucket_ids = self.dataset.data.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=1, #always train in original fps
            seed=self.seed, #TODO: 如果可能，每个epoch更换随机种子
            num_bucket=self.bucket.num_bucket,
            train_fps = self.train_fps
        )
        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        real_thw = {}
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            real_thw[i] = (real_t, real_h, real_w)
        self.real_thw = real_thw
        return bucket_sample_dict

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if self.verbose:
            print("Bucket Info:")
            print(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            print(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            print(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            print(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def state_dict(self, num_steps: int) -> dict:
        #TODO: support resume dataloader
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

