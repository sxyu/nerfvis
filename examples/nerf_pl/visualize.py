import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import imageio
import nerfvis
from datasets import dataset_dict
from datasets.depth_utils import *
from models.nerf import *
from models.rendering import render_rays
from utils import load_ckpt

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=os.path.join(dir_path, 'silica'),
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='silica',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504, 378],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=True, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default=os.path.join(dir_path, 'silica/silica.ckpt'),
                        help='pretrained checkpoint path to load')
    parser.add_argument('--port', type=int, default=8889,
                        help='port to run server')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    scene = nerfvis.Scene(args.scene_name)

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)

    nerf_fine = NeRF()
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval()

    @torch.no_grad()
    def nerf_func(points, dirs):
        # points [B, 1, 3]
        # dirs [1, sh_proj_sample_count, 3]
        xyz_embedded = embedding_xyz(points)
        dir_embedded = embedding_dir(dirs)
        xyzdir_embedded = torch.cat([xyz_embedded.expand(-1, dirs.size(1), -1),
                                     dir_embedded.expand(points.size(0), -1, -1)], -1)
        result = nerf_fine(xyzdir_embedded)
        return result[..., :3], result[..., 3:]


    # This will project to SH. You can change sh_deg to use higher-degree SH 
    # or increase sh_proj_sample_count to improve the accuracy of the projection
    scene.set_nerf(nerf_func, center=[0.0, 0.0, -1.5], radius=0.9, use_dirs=True)
    scene.display(port=args.port)
