from torchvision.transforms import Compose
from transformers import AutoTokenizer, AutoImageProcessor

from torchvision import transforms
from torchvision.transforms import Lambda

from opensora.dataset.t2v_multires_datasets import T2VMultiRes_dataset
from opensora.dataset.bodydance_dataset_3 import BodyDance_dataset_3
from opensora.dataset.bodydance_dataset_refmask import BodyDance_dataset as BodyDance_dataset_RefMask
from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo
# from opensora.models.causalvideovae import ae_norm, ae_denorm

ae_norm = Lambda(lambda x: 2. * x - 1.)
ae_denorm = lambda x: (x + 1.) / 2.

def getdataset(args):
    temporal_sample = TemporalRandomCrop()  # 16 x 简单地时序上的随机采样
    norm_fun = ae_norm
    if args.dataset == 't2v':
        pass

    elif args.dataset == 't2v_multi_res':
        #TODO: 优化下
        return T2VMultiRes_dataset(args, transform=None, temporal_sample=temporal_sample, tokenizer=None, 
                           transform_topcrop=None, valid=args.valid, 
                           ignore_timestamps=args.ignore_timestamps, one_more_second=args.one_more_second, temporal_downscale=args.temporal_downscale,
                           load_cached_latents=args.load_cached_latents,
                           null_embedding_path=args.null_embedding_path
                           )
    elif args.dataset == 'bodydance_3':
        return BodyDance_dataset_3(
            args, transform=None, temporal_sample=temporal_sample, tokenizer=None, 
                           transform_topcrop=None, valid=args.valid, 
                           ignore_timestamps=args.ignore_timestamps, one_more_second=args.one_more_second, temporal_downscale=args.temporal_downscale,
                           load_cached_latents=args.load_cached_latents,
                           null_embedding_path=args.null_embedding_path
        )
    elif args.dataset == 'bodydance_refmask':
        return BodyDance_dataset_RefMask(
            args, transform=None, temporal_sample=temporal_sample, tokenizer=None, 
                           transform_topcrop=None, valid=args.valid, 
                           ignore_timestamps=args.ignore_timestamps, one_more_second=args.one_more_second, temporal_downscale=args.temporal_downscale,
                           load_cached_latents=args.load_cached_latents,
                           null_embedding_path=args.null_embedding_path
        )

    raise NotImplementedError(args.dataset)


if __name__ == "__main__":
    from accelerate import Accelerator
    from opensora.dataset.t2v_datasets import dataset_prog
    import random
    from tqdm import tqdm
    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'dataset': 't2v', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 320,
        'max_width': 240,
        'num_frames': 1,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        'cache_dir': '../cache_dir', 
        'image_data': '/storage/ongoing/new/Open-Sora-Plan-bak/7.14bak/scripts/train_data/image_data.txt', 
        'video_data': '1',
        'train_fps': 24, 
        'drop_short_ratio': 1.0, 
        'use_img_from_vid': False, 
        'speed_factor': 1.0, 
        'cfg': 0.1, 
        'text_encoder_name': 'google/mt5-xxl', 
        'dataloader_num_workers': 10,

    }
    )
    accelerator = Accelerator()
    dataset = getdataset(args)
    num = len(dataset_prog.img_cap_list)
    zero = 0
    for idx in tqdm(range(num)):
        image_data = dataset_prog.img_cap_list[idx]
        caps = [i['cap'] if isinstance(i['cap'], list) else [i['cap']] for i in image_data]
        try:
            caps = [[random.choice(i)] for i in caps]
        except Exception as e:
            print(e)
            # import ipdb;ipdb.set_trace()
            print(image_data)
            zero += 1
            continue
        assert caps[0] is not None and len(caps[0]) > 0
    print(num, zero)
    import ipdb;ipdb.set_trace()
    print('end')