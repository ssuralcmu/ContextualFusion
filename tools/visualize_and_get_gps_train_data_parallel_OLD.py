import argparse
import copy
import os
import pyproj
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            obj[key] = recursive_eval(obj[key], globals)

    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def process_sample(data_and_args):
    data, args, cfg, model = data_and_args
    
    try:
        # Reinitialize NuScenes in each worker
        nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=False)
        
        # Existing processing logic
        metas = data["metas"].data[0][0]
        token = metas['token']
        sample = nusc.get('sample', token)
        scene_token = sample['scene_token']
        scene = nusc.get('scene', scene_token)
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']

        nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name=map_name)

        sample = nusc.get('sample', metas['token'])
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        global_xyz = ego_pose['translation']

        patch_size = 200

        fig, ax = nusc_map.render_map_patch(
            box_coords=[global_xyz[0]-patch_size, global_xyz[1]-patch_size, global_xyz[0]+patch_size, global_xyz[1]+patch_size],
            figsize=(10, 10),
            layer_names=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane', 'road_segment', 'road_block'],
            render_egoposes_range=False,
            render_legend=False,
            alpha=0.5
        )

        name = "{}-{}".format(metas["timestamp"], metas["token"])

        save_path = f'all_{args.split}_basemaps/{name}_based_map_image.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        metas_save_path = f'all_{args.split}_metas/{name}_metas.npy'
        np.save(metas_save_path, metas)

        # Additional processing logic
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

            if "masks_bev" in outputs[0]:
                masks = outputs[0]["masks_bev"].numpy()
                masks = masks >= args.map_score
            else:
                masks = None
        elif args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        else:
            masks = None

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_generated_map_image.png"),
                masks,
                classes=cfg.map_classes,
            )
        return True
    except Exception as e:
        print(f"Failed processing {token}: {str(e)}")
        return False
    except KeyboardInterrupt:
        print("Processing interrupted.")
        return False
def make_pickleable(item):
    # If it's a dict, process both keys and values.
    if isinstance(item, dict):
        return {make_pickleable(k): make_pickleable(v) for k, v in item.items()}
    # Process lists recursively.
    elif isinstance(item, list):
        return [make_pickleable(x) for x in item]
    # Process tuples recursively.
    elif isinstance(item, tuple):
        return tuple(make_pickleable(x) for x in item)
    # Process sets recursively.
    elif isinstance(item, set):
        return {make_pickleable(x) for x in item}
    # Convert dictionary view types (dict_keys, dict_values) to lists.
    elif isinstance(item, type({}.keys())):
        return list(item)
    # Otherwise, assume the item is already pickleable.
    else:
        return item

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    np.set_printoptions(suppress=True)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,#cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    model = None
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    global nusc
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=True)

    # Prepare data for multiprocessing
    data_with_args = [
        (make_pickleable(data), args, make_pickleable(cfg), model)
        for data in dataflow
    ]

    # Use multiprocessing to parallelize the processing
    with Pool(processes=512) as pool:
        chunksize = max(1, len(dataflow) // (4 * cpu_count()))  # Better for large datasets
        list(tqdm(pool.imap_unordered(process_sample, data_with_args, chunksize=chunksize), total=len(dataflow)))

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Add this line
    main()

