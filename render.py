#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    tag = globals().get("TAG", "")
    sub = name + (f"_{tag}" if tag else "")
    render_path = os.path.join(model_path, sub, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, sub, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Load data (deserialize)
    smpl_rot = {}
    rot_file = model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle'
    if os.path.exists(rot_file):
        with open(rot_file, 'rb') as handle:
            smpl_rot = pickle.load(handle)
    FRESH = globals().get("FRESH", False) or bool(tag)   # a tagged split = new frames -> fresh transforms

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask
        # [vpskin] reuse the transforms cached at training time (GauHuman protocol); frames that were
        # never seen in training (novel-pose splits) or --fresh_transforms compute them from the pose
        cached = None if FRESH else smpl_rot.get(name, {}).get(view.pose_id)
        if cached is not None:
            transforms, translation = cached['transforms'], cached['translation']
            transforms_cov = cached.get('transforms_cov', transforms)
        else:
            transforms = translation = transforms_cov = None

        # Start timer
        start_time = time.time() 
        render_output = render(view, gaussians, pipeline, background, transforms=transforms, translation=translation, transforms_cov=transforms_cov)
        rendering = render_output["render"]
        
        # end time  [vpskin] synchronize so FPS measures GPU work, not kernel launch
        torch.cuda.synchronize()
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)

    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))

        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

    # [vpskin] persist metrics (the metric code above is GauHuman's, unchanged)
    import json
    out = {"split": name, "iteration": iteration, "n_views": len(views), "psnr": float(psnrs),
           "ssim": float(ssims), "lpips_vgg": float(lpipss), "lpips_x1000": 1000 * float(lpipss),
           "fps_sync": len(views) / elapsed_time, "n_gaussians": int(gaussians.get_xyz.shape[0])}
    out["tag"] = tag
    with open(os.path.join(model_path, f"metrics_{sub}_{iteration}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender, skin_mode=getattr(dataset, 'skin_mode', 'lbs'), beta_init=getattr(dataset, 'beta_init', 0.0))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fresh_transforms", action="store_true", help="[vpskin] ignore cached smpl_rot transforms")
    parser.add_argument("--tag", type=str, default="", help="[vpskin] suffix for the metrics json / render folder")
    args = get_combined_args(parser)
    # a different split file means different frames: never reuse cached per-frame transforms
    FRESH = bool(args.fresh_transforms) or bool(getattr(args, "split_file", ""))
    TAG = args.tag or (os.path.splitext(os.path.basename(args.split_file))[0] if getattr(args, "split_file", "") else "")
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # [vpskin] log peak VRAM / wall-clock for evaluation too
    from utils.runlog import RunLog
    with RunLog("render:" + args.model_path, config=vars(args)):
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)