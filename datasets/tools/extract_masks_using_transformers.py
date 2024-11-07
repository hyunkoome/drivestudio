from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
from PIL import Image
from argparse import ArgumentParser
import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.nn import functional as F

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],  # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18],  # 'person', 'rider', 'motorcycle', 'bicycle'
    'sky': [10],  # 'sky'
}


class Segmentor():
    def __init__(self, model_name: str = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'):
        self.valid_model_names = [
            'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b5-finetuned-ade-640-640',
            'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b4-finetuned-ade-512-512',
            'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b3-finetuned-ade-512-512',
            'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b2-finetuned-ade-512-512',
            'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b1-finetuned-ade-512-512',
            'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b0-finetuned-cityscapes-512-1024',
            'nvidia/segformer-b0-finetuned-cityscapes-640-1280',
            'nvidia/segformer-b0-finetuned-cityscapes-768-768',
            'nvidia/segformer-b0-finetuned-ade-512-512'
        ]

        assert model_name in self.valid_model_names, f"Invalid model_name: {model_name}. Please choose from {self.valid_model_names}"

        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    def inference(self, image_file_path: str) -> np.ndarray:
        """
        Args:
            image_file_path: str
        """

        image = Image.open(image_file_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # model inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # extract class-wise logits: Segmentation Mask
        pred = outputs.logits

        # logits size -> interpolate -> original image size
        result = F.interpolate(input=pred, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
        mask = torch.argmax(result, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask


class SegmentorBatch():
    def __init__(self, model_name: str = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024', device: str = 'cuda:0'):
        self.valid_model_names = [
            'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b5-finetuned-ade-640-640',
            'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b4-finetuned-ade-512-512',
            'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b3-finetuned-ade-512-512',
            'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b2-finetuned-ade-512-512',
            'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b1-finetuned-ade-512-512',
            'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
            'nvidia/segformer-b0-finetuned-cityscapes-512-1024',
            'nvidia/segformer-b0-finetuned-cityscapes-640-1280',
            'nvidia/segformer-b0-finetuned-cityscapes-768-768',
            'nvidia/segformer-b0-finetuned-ade-512-512'
        ]

        assert model_name in self.valid_model_names, f"Invalid model_name: {model_name}. Please choose from {self.valid_model_names}"

        # Load feature extractor and model
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)

    def batch_inference(self, image_file_paths: list) -> list:
        """
        Perform inference on a batch of images using SegformerFeatureExtractor.

        Args:
            image_file_paths: List of image file paths.

        Returns:
            List of segmentation masks (numpy arrays).
        """
        # Load and preprocess images using SegformerFeatureExtractor
        images = [Image.open(path).convert("RGB") for path in image_file_paths]
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract class-wise logits and resize to original image size
        logits = outputs.logits  # Shape: (batch_size, num_classes, height/4, width/4)
        masks = []
        for i, img in enumerate(images):
            original_size = img.size[::-1]  # Original size (height, width)
            upscaled_logits = F.interpolate(
                logits[i].unsqueeze(0),
                size=original_size,
                mode="bilinear",
                align_corners=False
            )
            mask = torch.argmax(upscaled_logits, dim=1).squeeze(0).cpu().numpy()
            masks.append(mask)
        return masks


def cleanAll():
    import torch

    torch.cuda.empty_cache()  # 캐시 비우기
    torch.cuda.memory_summary(device=None, abbreviated=False)  # 메모리 상태 출력

    import tensorflow as tf
    tf.keras.backend.clear_session()  # 모든 세션 정리


if __name__ == "__main__":
    cleanAll()

    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default='data/waymo_example_scenes.txt', help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        '--process_dynamic_mask',
        action='store_true',
        help="Whether to process dynamic masks",
    )
    # parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    # parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for processing images")

    # Algorithm configs
    # parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    # parser.add_argument('--config', help='Config file', type=str, default=None)
    # parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--model_name', help='model name of transformers, SegformerForSemanticSegmentation', type=str,
                        default='nvidia/segformer-b5-finetuned-cityscapes-1024-1024')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')

    args = parser.parse_args()

    # split_file = open(args.split_file, "r").readlines()[1:]

    # if args.config is None:
    #     args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5',
    #                                'segformer.b5.1024x1024.city.160k.py')
    # if args.checkpoint is None:
    #     args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        # NOTE: small hack here, to be refined in the futher (TODO)
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    # 1. load feature extractor and model using transformers and hugging face
    # seg_model = Segmentor(model_name=args.model_name)
    seg_model = SegmentorBatch(model_name=args.model_name, device=args.device)

    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, f'Extracting Masks ...')):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        # create mask dir
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        if not os.path.exists(sky_mask_dir):
            os.makedirs(sky_mask_dir)

        # create dynamic mask dir
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")

            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            if not os.path.exists(all_mask_dir):
                os.makedirs(all_mask_dir)
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            if not os.path.exists(human_mask_dir):
                os.makedirs(human_mask_dir)
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            if not os.path.exists(vehicle_mask_dir):
                os.makedirs(vehicle_mask_dir)

        flist = sorted(glob(os.path.join(img_dir, '*')))
        # for fpath in tqdm(flist, f'scene[{scene_id}]'):
        #     fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
        for batch_start in tqdm(range(0, len(flist), args.batch_size), desc=f"Scene[{scene_id}]"):
            batch_files = flist[batch_start:batch_start + args.batch_size]

            # if args.no_compress:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npy")
            # else:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npz")

            if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                continue

            # code updated by hyunkoo kim ... start
            masks = seg_model.batch_inference(image_file_paths=batch_files)

            for i, mask in enumerate(masks):
                fbase = os.path.splitext(os.path.basename(batch_files[i]))[0]

                # # Save sky mask
                # sky_mask = np.isin(mask, dataset_classes_in_sematic['sky'])
                # imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

                # if args.no_compress:
                #     np.save(mask_fpath, mask)
                # else:
                #     np.savez_compressed(mask_fpath, mask)   # NOTE: compressed files are 100x smaller.

                # save sky mask
                # sky_mask = np.isin(mask, [10])
                sky_mask = np.isin(mask, dataset_classes_in_sematic['sky'])
                imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

                if args.process_dynamic_mask:
                    # save human masks
                    rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                    rough_human_mask = (imageio.imread(rough_human_mask_path) > 0)
                    huamn_mask = np.isin(mask, dataset_classes_in_sematic['human'])
                    valid_human_mask = np.logical_and(huamn_mask, rough_human_mask)
                    imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"),
                                    valid_human_mask.astype(np.uint8) * 255)

                    # save vehicle mask
                    rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")
                    rough_vehicle_mask = (imageio.imread(rough_vehicle_mask_path) > 0)
                    vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                    valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                    imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"),
                                    valid_vehicle_mask.astype(np.uint8) * 255)

                    # save dynamic mask
                    valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                    imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)
