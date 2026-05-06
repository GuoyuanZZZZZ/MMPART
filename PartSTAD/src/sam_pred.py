import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
sys.path.append("./segment-anything")
sys.path.append("../mm-sam-main")
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm, trange
import glob
import time
import os
import argparse

import imageio as io



class MMSAMAdapter:
    """Compatibility adapter that keeps PartSTAD SAM call-site unchanged."""

    def __init__(self, model_id: str, model_type: str = "wmmf", input_modality: str = "rgbd", device: str = "cuda"):
        try:
            from mm_sam.models.sam import SAMbyUCMT, SAMbyWMMF
        except ImportError as exc:
            raise ImportError(
                "Cannot import mm_sam. Please install mm-sam dependencies and ensure mm-sam-main is available."
            ) from exc

        self.model_type = model_type
        self.input_modality = input_modality
        model_cls = SAMbyWMMF if model_type == "wmmf" else SAMbyUCMT
        self.model = model_cls.from_pretrained(model_id).to(device).eval()
        self.device = device

    def set_image(self, image_rgb: np.ndarray, image_depth: np.ndarray = None):
        # MMSAM expects HWC tensors with channel_last=True
        if self.model_type == "wmmf" or self.input_modality == "rgbd":
            if image_depth is None:
                raise ValueError("Depth image is required for WMMF/rgbd inference.")
            if image_depth.ndim == 2:
                image_depth = image_depth[..., None]
            self.model.set_infer_img(rgb_img=image_rgb, x_img=image_depth, channel_last=True)
        elif self.input_modality == "depth":
            if image_depth is None:
                raise ValueError("Depth image is required for depth-only inference.")
            if image_depth.ndim == 2:
                image_depth = image_depth[..., None]
            self.model.set_infer_img(img=image_depth, channel_last=True)
        else:
            self.model.set_infer_img(img=image_rgb, channel_last=True)

    def predict_boxes(self, boxes_xyxy: torch.Tensor):
        # Keep shape convention with original SAM predictor: (N, 1, H, W)
        pred_masks, _ = self.model.infer(box_coords=boxes_xyxy, return_all_prompt_masks=True)
        mask = pred_masks[0]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        return mask
all_categories = ["Bottle","Box","Bucket","Camera","Cart","Clock","CoffeeMachine","Dishwasher","Dispenser","Display","Door","Eyeglasses","Faucet","FoldingChair","Globe","Keyboard","KitchenPot","Knife","Laptop","Lighter","Microwave","Mouse","Oven","Pen","Phone","Pliers","Printer","Remote","Safe","Scissors","Stapler","Switch","Toaster","Toilet","TrashCan","USB","WashingMachine","Window","Kettle","Lamp","Refrigerator","Suitcase","StorageFurniture",'Table','Chair']


print(len(all_categories))
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def draw_rectangle(img, x0, y0, x1, y1, c=None):
    if c is None:
        color = np.random.rand(3) * 255
    else:
        color = np.array(c[:3])*255
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    img[y0:y1, x0:x1, :3] /= 2
    img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img

cmap = plt.get_cmap("turbo")

def sam_init(args=None):
    device = getattr(args, "device", "cuda") if args is not None else "cuda"
    use_mmsam = getattr(args, "use_mmsam", False) if args is not None else False

    if use_mmsam:
        mmsam_model_id = getattr(args, "mmsam_model_id", "weihao1115/ucmt_sam_on_depth")
        return MMSAMAdapter(model_id=mmsam_model_id, model_type=getattr(args, "mmsam_model_type", "wmmf"), input_modality=getattr(args, "mmsam_input_modality", "rgbd"), device=device)

    sam_checkpoint = getattr(args, "sam_checkpoint", "sam_vit_h_4b8939.pth")
    model_type = getattr(args, "sam_model_type", "vit_h")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def sam_predict(args, meta, category, predictor=None):
    part_names = meta[category]
    num_label = len(part_names)

    if predictor is None:
        predictor = sam_init(args)
    device = getattr(args, "device", "cuda")

    pre_dir = args.preprocess_dir
    pc_dirs = sorted(glob.glob(f"{args.data_dir}/{category}/*"))
    save_dir = args.mask_dir
    glip_dir = args.glip_dir
    os.makedirs(save_dir, exist_ok=True)

    img_dir = "rendered_img"
    meta_dir = "img_meta"


    num_object = len(pc_dirs)
    pbar = trange(num_object)
    for i in pbar:
        pc_dir = pc_dirs[i]
        fname = pc_dir.split("/")[-1]
        # fsave_dir = f"{save_dir}/{fname}"
        fpre_dir = f"{pre_dir}/{category}/{fname}"
        fsave_dir = f"{save_dir}/{category}/{fname}"
        fglip_dir = f"{glip_dir}/{category}/{fname}"
        os.makedirs(fsave_dir, exist_ok=True)


        fimg_dir = f"{fpre_dir}/{img_dir}"

        img_paths = sorted(glob.glob(f"{fimg_dir}/*.png"))
        sam_save_dir = f"{fsave_dir}/SAM_result_img"
        os.makedirs(sam_save_dir,exist_ok=True)
        
        with open(f"{fglip_dir}/glip_pred/pred.json", "r") as f:
            glip_pred = json.load(f)

        bbox_masks = np.zeros((len(glip_pred),800,800))
        
        start_view = 0
        num_views = 10
        
        pbar.set_description("SAM prediction")
        for j in range(start_view, num_views):
            
            img_path = f"{fimg_dir}/{j}.png"

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H,W,_ = img.shape

            boxes = []
            bbox_idxs = []
            for bidx, pred in enumerate(glip_pred):
                if pred['image_id'] != j:
                    continue
                x,y,w,h = pred['bbox']
                x0,y0,x1,y1 = x,y,x+w,y+h
                box = torch.Tensor([x0,y0,x1,y1])
                boxes.append(box)
                bbox_idxs.append(bidx)

            if len(boxes) == 0:
                continue
            
            boxes = torch.stack(boxes,0)
            boxes_torch = boxes.to(device)

            if isinstance(predictor, MMSAMAdapter):
                depth = None
                depth_path = f"{fpre_dir}/rendered_depth/{j}.png"
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                predictor.set_image(img, depth)
                masks = predictor.predict_boxes(boxes_torch)
            else:
                predictor.set_image(img)
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_torch, img.shape[:2])
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

            masks = masks.detach().cpu().numpy()[:,0] # N,H,W
            num_masks = len(masks)

            boxes = boxes.numpy()

            for k in range(num_masks):
                seg = masks[k]
                bbox_idx = bbox_idxs[k]
                bbox_masks[bbox_idx] = seg

                seg_ori = np.uint8(seg*255)
                seg_as_img = 255*np.stack([seg,seg,seg],-1) #H,W,3

                merged = img*0.3 + seg_as_img*0.7
                merged = np.uint8(merged)

                x0,y0,x1,y1 = boxes[k]
                merged = draw_rectangle(merged,int(x0),int(y0),int(x1),int(y1),c=[1,0,0])

                io.imsave(f"{sam_save_dir}/overlap_bbox{bbox_idx:03d}_view{j:02d}.png", merged)
                io.imsave(f"{sam_save_dir}/mask_bbox{bbox_idx:03d}_view{j:02d}.png", seg_ori)

        np.savez_compressed(f"{fsave_dir}/sam_mask.npz", mask=bbox_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default="Bottle", type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--data_dir', default='data/train', type=str)

    parser.add_argument('--preprocess_dir', default='rendered_pc', type=str)
    parser.add_argument('--mask_dir', default='sam_preprocess')

    parser.add_argument('--glip_dir', default='glip_preprocess/train')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--use_mmsam', action='store_true')
    parser.add_argument('--mmsam_model_type', default='wmmf', choices=['ucmt','wmmf'])
    parser.add_argument('--mmsam_input_modality', default='rgbd', choices=['rgb','depth','rgbd'])
    parser.add_argument('--mmsam_model_id', default='weihao1115/wmmf_sam_on_depth', type=str)
    parser.add_argument('--sam_checkpoint', default='sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--sam_model_type', default='vit_h', type=str)

    args = parser.parse_args()
    
    partnete_meta = json.load(open("PartNetE_meta.json"))
    category = args.category
    predictor = sam_init(args)
    if category == "all":
        print("Rendering For All Categories!")
        for category in tqdm(all_categories[args.start:args.end]):
            print(f"Category : {category}")
            sam_predict(args, partnete_meta, category, predictor)
    else:
        sam_predict(args, partnete_meta, category, predictor)


