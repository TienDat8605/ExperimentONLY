import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger

import re
from PIL import Image

from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

# NLTK for CHAIR lemmatization
try:
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None

# Download wordnet data if needed
import nltk
try:
    nltk.data.find('corpora/wordnet.zip') or nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--chair_objects_path", type=str, default="data/chair/coco_objects.json")
    parser.add_argument("--chair_captions_path", type=str, default="data/chair/captions_val2014.json")
    parser.add_argument("--log_path", type=str, default="logs/chair")
    parser.add_argument("--captions_output", type=str, default=None,
                        help="Optional JSONL path consumed by the official CHAIR scorer.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    parser.add_argument("--mask_alpha", type=float, default=0.2)

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_images", type=int, default=0,
                        help="If >0, stop after this many images (for quick iteration).")
    parser.add_argument("--proposal", type=int, default=1)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--score_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_decay", type=float, default=0.3)
    parser.add_argument("--debug_tvd", type=str2bool, default=False)
    parser.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.25)
    parser.add_argument("--expert_layers", type=str, default="0,8,16,24")
    parser.add_argument("--consensus_min", type=float, default=0.75)
    parser.add_argument("--consensus_strength", type=float, default=1.0)
    parser.add_argument("--entropy_temperature", type=float, default=1.0)

    args = parser.parse_args()
    return args


class COCOCaptionDataset(Dataset):
    """Iterate over COCO val2014 images for caption generation."""

    def __init__(self, data_path, captions_path, trans, max_images=0):
        self.data_path = data_path
        self.trans = trans

        # Load captions_val2014.json
        with open(captions_path, 'r') as f:
            coco_data = json.load(f)

        # Build image_id -> file_name mapping
        self.img_id_to_file = {}
        for img_info in coco_data['images']:
            self.img_id_to_file[img_info['id']] = img_info['file_name']

        # Build image_id -> [caption, ...] mapping (GT captions)
        self.img_id_to_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_captions:
                self.img_id_to_captions[img_id] = []
            self.img_id_to_captions[img_id].append(ann['caption'])

        # Sorted list of all image IDs with captions
        all_ids = sorted(self.img_id_to_captions.keys())
        if max_images > 0:
            all_ids = all_ids[:max_images]

        self.image_ids = all_ids
        self.n_images = len(self.image_ids)
        print(f"[CHAIR Dataset] {self.n_images} images loaded from {captions_path}")

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        file_name = self.img_id_to_file[img_id]
        image_path = os.path.join(self.data_path, file_name)

        raw_image = Image.open(image_path).convert('RGB')
        image = self.trans.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        return {
            "image": image,
            "image_id": img_id,
            "image_path": image_path,
            "gt_captions": self.img_id_to_captions[img_id],
            "file_name": file_name,
        }


def load_coco_objects(objects_path):
    """Load coco_objects.json: {base_object: [synonyms]}."""
    with open(objects_path, 'r') as f:
        return json.load(f)


def extract_objects(text, coco_objects):
    """Extract COCO object names from text via lemmatization + synonym matching.

    Handles multi-word objects (e.g. 'dining table', 'cell phone') by
    checking bigrams as well as unigrams.

    Returns list of base object names found in the text.
    """
    # Tokenize, lowercase, lemmatize
    text = text.lower().replace('.', ' ').replace(',', ' ').replace(';', ' ').replace('?', ' ').replace('!', ' ')
    words = text.split()
    lemmatized = set()
    for w in words:
        lemmatized.add(_lemmatizer.lemmatize(w))
        if w.endswith('ing'):
            lemmatized.add(_lemmatizer.lemmatize(w, 'v'))  # try verb lemmatization

    # Also build bigram set
    bigrams = set()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")
        bigrams.add(f"{_lemmatizer.lemmatize(words[i])} {_lemmatizer.lemmatize(words[i+1])}")

    found = []
    for base_obj, synonyms in coco_objects.items():
        syn_lower = [s.lower() for s in synonyms]
        # Check unigrams
        if any(syn in lemmatized for syn in syn_lower):
            found.append(base_obj)
            continue
        # Check bigrams for multi-word objects
        if any(syn in bigrams for syn in syn_lower):
            found.append(base_obj)
            continue

    return found


def compute_chair(generated_captions, coco_objects, gt_captions_per_img, logger):
    """Compute CHAIR_I (instance-level) and CHAIR_S (sentence-level).

    Args:
        generated_captions: dict {image_id: caption_string}
        coco_objects: dict {base_object: [synonyms]}
        gt_captions_per_img: dict {image_id: [caption1, ...]}
    Returns:
        (CHAIR_I, CHAIR_S, details)
    """
    logger.info("Computing captions-only diagnostic (official CHAIR runs separately)...")

    # Build GT object set per image from reference captions
    logger.info("Building ground-truth object sets from reference captions...")
    gt_objects_per_image = {}
    for img_id, captions in gt_captions_per_img.items():
        objs = set()
        for cap in captions:
            cap_objs = extract_objects(cap, coco_objects)
            objs.update(cap_objs)
        gt_objects_per_image[img_id] = objs

    # Score generated captions
    total_mentioned = 0
    total_hallucinated = 0
    total_sentences = 0
    hallucinated_sentences = 0

    # Per-object statistics
    per_obj_gt_counts = {}   # how many images have this object in GT
    per_obj_gen_counts = {}  # how many images have this object in generated caption
    per_obj_hallu_counts = {}  # how many times this object was hallucinated

    for img_id, caption in generated_captions.items():
        gen_objs = extract_objects(caption, coco_objects)
        gt_objs = gt_objects_per_image.get(img_id, set())

        # Count generated objects
        for obj in gen_objs:
            per_obj_gen_counts[obj] = per_obj_gen_counts.get(obj, 0) + 1
            if obj not in gt_objs:
                total_hallucinated += 1
                per_obj_hallu_counts[obj] = per_obj_hallu_counts.get(obj, 0) + 1
            else:
                # Count as a non-hallucinated mention
                pass  # already counted in total_mentioned

        # Count GT objects
        for obj in gt_objs:
            per_obj_gt_counts[obj] = per_obj_gt_counts.get(obj, 0) + 1

        total_mentioned += len(gen_objs)
        total_sentences += 1
        if any(obj not in gt_objs for obj in gen_objs):
            hallucinated_sentences += 1

    chair_i = total_hallucinated / max(total_mentioned, 1)
    chair_s = hallucinated_sentences / max(total_sentences, 1)

    details = {
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
        "total_sentences": total_sentences,
        "hallucinated_sentences": hallucinated_sentences,
        "per_object_gt": per_obj_gt_counts,
        "per_object_gen": per_obj_gen_counts,
        "per_object_hallu": per_obj_hallu_counts,
    }

    # Log per-object details
    logger.info("Per-object statistics:")
    sorted_objects = sorted(set(
        list(per_obj_gt_counts.keys()) + list(per_obj_gen_counts.keys())
    ))
    header = f"{'Object':<20} {'GT_imgs':>10} {'Gen_imgs':>10} {'Hallu':>10} {'Hallu_rate':>12}"
    logger.info(header)
    logger.info("-" * len(header))
    for obj in sorted_objects:
        gt_c = per_obj_gt_counts.get(obj, 0)
        gen_c = per_obj_gen_counts.get(obj, 0)
        hallu_c = per_obj_hallu_counts.get(obj, 0)
        hallu_rate = hallu_c / max(gen_c, 1) * 100
        logger.info(f"{obj:<20} {gt_c:>10} {gen_c:>10} {hallu_c:>10} {hallu_rate:>11.1f}%")

    return chair_i, chair_s, details


def main():
    args = parse_args()

    # Setup DDP
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        method_name = "ONLY" if args.use_only else "Regular"
        experiment_dir = os.path.join(
            args.log_path,
            f"{model_string_name}/{method_name}_chair_proposal{args.proposal}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    # Load COCO objects for CHAIR scoring
    logger.info(f"Loading COCO objects from {args.chair_objects_path}")
    coco_objects = load_coco_objects(args.chair_objects_path)
    logger.info(f"Loaded {len(coco_objects)} COCO object categories")

    # Load GT captions for CHAIR
    logger.info(f"Loading reference captions from {args.chair_captions_path}")
    with open(args.chair_captions_path, 'r') as f:
        captions_data = json.load(f)

    # Build GT captions dict: image_id -> [captions]
    gt_captions_per_img = {}
    for ann in captions_data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_captions_per_img:
            gt_captions_per_img[img_id] = []
        gt_captions_per_img[img_id].append(ann['caption'])

    img_id_to_file = {}
    for img_info in captions_data['images']:
        img_id_to_file[img_info['id']] = img_info['file_name']

    all_ids = sorted(gt_captions_per_img.keys())
    random.Random(args.seed).shuffle(all_ids)
    if args.max_images > 0:
        all_ids = all_ids[:args.max_images]

    logger.info(f"Total images with captions: {len(all_ids)}")

    # ==============================================
    #            Start Generation
    # ==============================================
    logger.info("Start caption generation for CHAIR evaluation...")
    generated_captions = {}  # image_id -> caption string

    for idx, img_id in enumerate(tqdm(all_ids)):
        file_name = img_id_to_file[img_id]
        image_path = os.path.join(args.data_path, file_name)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}, skipping")
            continue

        # Load and preprocess image
        raw_image = Image.open(image_path).convert('RGB')
        image = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        # Build prompt: "<image>\nDescribe this image in detail."
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )

        prompt_text = "Describe this image in detail."
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        with torch.inference_mode():
            with torch.no_grad():
                output_ids, _ = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    images_pos=None,
                    images_neg=None,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    use_ritual=False,
                    use_vcd=False,
                    use_m3id=False,
                    use_only=args.use_only,
                    enhance_layer_index=args.enhance_layer_index,
                    mask_alpha=args.mask_alpha,
                    debug_tvd=args.debug_tvd,
                    proposal=args.proposal,
                    score_threshold=args.score_threshold,
                    score_temperature=args.score_temperature,
                    lambda_decay=args.lambda_decay,
                    ritual_alpha_pos=args.ritual_alpha_pos,
                    ritual_alpha_neg=args.ritual_alpha_neg,
                    ritual_beta=args.ritual_beta,
                    js_gamma=args.js_gamma,
                    expert_layers=args.expert_layers,
                    consensus_min=args.consensus_min,
                    consensus_strength=args.consensus_strength,
                    entropy_temperature=args.entropy_temperature,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids differ from input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        generated_captions[img_id] = outputs

        if idx % 100 == 0:
            logger.info(f"[{idx}/{len(all_ids)}] Generated caption for image_id={img_id}")
            logger.info(f"  Caption: {outputs[:120]}...")

    logger.info(f"Generated {len(generated_captions)} captions total")

    if args.captions_output:
        output_dir = os.path.dirname(os.path.abspath(args.captions_output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.captions_output, "w") as f:
            for image_id, caption in generated_captions.items():
                f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
        logger.info(f"Official-CHAIR caption input saved to {args.captions_output}")

    # ==============================================
    #            Compute CHAIR Scores
    # ==============================================
    chair_i, chair_s, details = compute_chair(
        generated_captions, coco_objects, gt_captions_per_img, logger
    )

    logger.info("=" * 60)
    logger.info("Captions-only diagnostic (NOT official CHAIR)")
    logger.info("=" * 60)
    logger.info(f"CHAIR_I (Instance-level hallucination rate): {chair_i:.4f} ({chair_i*100:.2f}%)")
    logger.info(f"CHAIR_S (Sentence-level hallucination rate): {chair_s:.4f} ({chair_s*100:.2f}%)")
    logger.info(f"Total mentioned objects: {details['total_mentioned']}")
    logger.info(f"Total hallucinated instances: {details['total_hallucinated']}")
    logger.info(f"Total sentences: {details['total_sentences']}")
    logger.info(f"Sentences with hallucination: {details['hallucinated_sentences']}")
    logger.info("=" * 60)

    # Also print for stdout
    print("\n" + "=" * 60)
    print("Captions-only diagnostic (NOT official CHAIR)")
    print("=" * 60)
    print(f"CHAIR_I: {chair_i:.4f} ({chair_i*100:.2f}%)")
    print(f"CHAIR_S: {chair_s:.4f} ({chair_s*100:.2f}%)")
    print(f"Total mentioned: {details['total_mentioned']}, Hallucinated: {details['total_hallucinated']}")
    print(f"Total sentences: {details['total_sentences']}, Hallu sentences: {details['hallucinated_sentences']}")
    print("=" * 60)

    # Log all args
    logger.info(vars(args))

    # Save results to file
    results_path = os.path.join(experiment_dir, "chair_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "chair_i": chair_i,
            "chair_s": chair_s,
            "chair_i_pct": chair_i * 100,
            "chair_s_pct": chair_s * 100,
            "total_mentioned": details['total_mentioned'],
            "total_hallucinated": details['total_hallucinated'],
            "total_sentences": details['total_sentences'],
            "hallucinated_sentences": details['hallucinated_sentences'],
            "args": vars(args),
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
