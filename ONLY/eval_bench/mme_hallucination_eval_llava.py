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
from torchvision.transforms import v2

from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')


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
    parser = argparse.ArgumentParser(description="MME-Hallucination evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="data/mme_hallucination")
    parser.add_argument("--mme_path", type=str, default="data/mme_hallucination/mme_hallucination.jsonl")
    parser.add_argument("--log_path", type=str, default="logs/mme_hallucination")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)
    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)
    parser.add_argument("--mask_alpha", type=float, default=0.2)

    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.6)

    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, default="mme_hallucination")
    parser.add_argument("--max_questions", type=int, default=0,
                        help="If >0, stop after this many questions (for quick iteration).")
    parser.add_argument("--debug_tvd", type=str2bool, default=False,
                        help="Log per-step TVD, regime, suppression_ratio, dynamic_alpha to stdout.")
    parser.add_argument("--proposal", type=int, default=1,
                        help="0=original ONLY (layer 0 only), 1=EMA mask (current/multi-layer), "
                             "2=score accumulation, 3=residual delta")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--score_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_decay", type=float, default=0.3)

    args = parser.parse_args()
    return args


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
        break
    return pred_list


class MMEHallucinationDataset(Dataset):
    """MME-Hallucination dataset from JSONL format.

    JSONL fields: question_id, image, text, label, category
    Images are stored in data_path/ directory.
    """

    def __init__(self, mme_path, data_path, trans, model, max_questions=0):
        self.data_path = data_path
        self.trans = trans
        self.model = model

        image_list, query_list, label_list, category_list = [], [], [], []

        with open(mme_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                image_list.append(rec['image'])
                query_list.append(rec['text'])
                # Normalize label: 'no' -> 0, else -> 1
                label = rec.get('label', rec.get('answer', 'yes'))
                if isinstance(label, str) and label.lower() == 'no':
                    label_list.append(0)
                else:
                    label_list.append(1)
                category_list.append(rec.get('category', 'Unknown'))

        if max_questions > 0:
            image_list = image_list[:max_questions]
            query_list = query_list[:max_questions]
            label_list = label_list[:max_questions]
            category_list = category_list[:max_questions]

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.category_list = category_list

        assert len(self.image_list) == len(self.query_list) == len(self.label_list) == len(self.category_list)

        # Gather unique categories
        self.categories = sorted(set(self.category_list))
        print(f"[MME-Hallucination] {len(self.image_list)} questions, "
              f"{len(self.categories)} categories: {self.categories}")

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])

        if self.model == 'llava':
            raw_image = Image.open(image_path).convert('RGB')
            image = self.trans.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]
        elif self.model == 'qwen-vl':
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans(raw_image)
        elif self.model == 'instructblip':
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans['eval'](raw_image)
        else:
            raw_image = Image.open(image_path).convert('RGB')
            image = self.trans.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]

        return {
            "image": image,
            "query": self.query_list[index],
            "label": self.label_list[index],
            "category": self.category_list[index],
            "image_path": image_path,
        }


def compute_metrics_by_category(results_by_category, logger):
    """Compute accuracy, precision, recall, F1 per category.

    results_by_category: dict {category: [(pred, gt_label), ...]}
    """
    metrics = {}
    overall_tp = overall_tn = overall_fp = overall_fn = 0

    logger.info("=" * 60)
    logger.info("MME-Hallucination Per-Category Results")
    logger.info("=" * 60)
    header = f"{'Category':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Count':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for category in sorted(results_by_category.keys()):
        results = results_by_category[category]
        tp = tn = fp = fn = 0
        for pred, gt in results:
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            elif pred == 0 and gt == 1:
                fn += 1

        overall_tp += tp
        overall_tn += tn
        overall_fp += fp
        overall_fn += fn

        total = tp + tn + fp + fn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        metrics[category] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "count": total,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

        logger.info(f"{category:<25} {acc*100:>7.2f}% {prec*100:>7.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}% {total:>8}")

    # Overall
    total_all = overall_tp + overall_tn + overall_fp + overall_fn
    overall_acc = (overall_tp + overall_tn) / max(total_all, 1)
    overall_prec = overall_tp / max(overall_tp + overall_fp, 1)
    overall_rec = overall_tp / max(overall_tp + overall_fn, 1)
    overall_f1 = 2 * overall_prec * overall_rec / max(overall_prec + overall_rec, 1e-8)

    logger.info("-" * len(header))
    logger.info(f"{'OVERALL':<25} {overall_acc*100:>7.2f}% {overall_prec*100:>7.2f}% "
                f"{overall_rec*100:>7.2f}% {overall_f1*100:>7.2f}% {total_all:>8}")

    metrics["overall"] = {
        "accuracy": round(overall_acc * 100, 2),
        "precision": round(overall_prec * 100, 2),
        "recall": round(overall_rec * 100, 2),
        "f1": round(overall_f1 * 100, 2),
        "count": total_all,
        "tp": overall_tp, "tn": overall_tn, "fp": overall_fp, "fn": overall_fn,
    }

    return metrics


def main():
    args = parse_args()

    # Setup DDP
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        model_string_name = args.model_path.split("/")[-1]
        if args.use_ritual:
            method_name = "RITUAL"
        elif args.use_vcd:
            method_name = "VCD"
        elif args.use_m3id:
            method_name = "M3ID"
        elif args.use_only:
            method_name = "ONLY"
        else:
            method_name = "Regular"
        experiment_dir = os.path.join(
            args.log_path,
            f"{model_string_name}/{method_name}_{args.dataset_name}"
            f"_{args.ritual_alpha_pos}_{args.ritual_alpha_neg}"
            f"_{args.ritual_beta}_{args.js_gamma}"
            f"_layer_{args.enhance_layer_index}_proposal{args.proposal}"
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

    # Load dataset
    mme_dataset = MMEHallucinationDataset(
        mme_path=args.mme_path,
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base,
        max_questions=args.max_questions,
    )
    mme_loader = torch.utils.data.DataLoader(
        mme_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ==============================================
    #               Augmentations
    # ==============================================
    aug_dict = {
        'horizontal flip': v2.RandomHorizontalFlip(p=1),
        'vertical flip': v2.RandomVerticalFlip(p=1),
        'rotation': v2.RandomRotation(degrees=180),
        'color jitter': v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
        'gaussian blur': v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
        'crop': v2.RandomResizedCrop(size=336),
    }
    pos_aug_counter = {k: 0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start MME-Hallucination evaluation...")
    pred_list, label_list = [], []
    category_list = []
    all_tvd_stats = []

    results_by_category = {}  # category -> list of (pred, gt)

    for batch_id, data in tqdm(enumerate(mme_loader), total=len(mme_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        category = data["category"][0]
        image_path = data["image_path"]
        label_list.extend(list(label))

        image_pos = None
        image_neg = None

        if args.use_ritual:
            raw_image = Image.open(image_path[0])
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0]
                image_pos = torch.tensor(image_pos)
            pos_aug_counter[pos_aug] += 1
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)

        # ==============================================
        #              Text prompt setting
        # ==============================================
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

        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        # ==============================================
        #                Generate
        # ==============================================
        with torch.inference_mode():
            with torch.no_grad():
                output_ids, overlapping_index_len = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda()
                                if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda()
                                if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    use_ritual=args.use_ritual,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    use_only=args.use_only,
                    enhance_layer_index=args.enhance_layer_index,
                    mask_alpha=args.mask_alpha,
                    ritual_alpha_pos=args.ritual_alpha_pos,
                    ritual_alpha_neg=args.ritual_alpha_neg,
                    ritual_beta=args.ritual_beta,
                    js_gamma=args.js_gamma,
                    debug_tvd=args.debug_tvd,
                    proposal=args.proposal,
                    score_threshold=args.score_threshold,
                    score_temperature=args.score_temperature,
                    lambda_decay=args.lambda_decay,
                )

        if args.debug_tvd:
            all_tvd_stats.extend(overlapping_index_len)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids differ from input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        pred_list = recorder(outputs, pred_list)
        print(f"[MME-Hallucination]")
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        print(f"A: {outputs}")
        print(f"GT: {'Yes' if label[0].item() == 1 else 'No'}")
        print(f"Category: {category}")

    # ==============================================
    #            Compute Results
    # ==============================================
    # Build per-category results
    for i in range(len(pred_list)):
        cat = mme_dataset.category_list[i]
        if cat not in results_by_category:
            results_by_category[cat] = []
        results_by_category[cat].append((pred_list[i], mme_dataset.label_list[i]))

    # Compute metrics
    metrics = compute_metrics_by_category(results_by_category, logger)

    # Log summary
    logger.info("=" * 60)
    logger.info("MME-Hallucination Summary")
    logger.info("=" * 60)
    logger.info(f"Total questions: {len(pred_list)}")
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2f}%")
    logger.info(f"Overall Precision: {metrics['overall']['precision']:.2f}%")
    logger.info(f"Overall Recall: {metrics['overall']['recall']:.2f}%")
    logger.info(f"Overall F1: {metrics['overall']['f1']:.2f}%")
    logger.info(vars(args))

    # Save results to JSON
    results_path = os.path.join(experiment_dir, "mme_hallucination_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "metrics": metrics,
            "total_questions": len(pred_list),
            "args": vars(args),
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Also print for stdout
    print("\n" + "=" * 60)
    print("MME-Hallucination Summary")
    print("=" * 60)
    print(f"Total questions: {len(pred_list)}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2f}%")
    print(f"Overall F1: {metrics['overall']['f1']:.2f}%")
    for cat, m in sorted(metrics.items()):
        if cat == "overall":
            continue
        print(f"  {cat}: Acc={m['accuracy']:.2f}% F1={m['f1']:.2f}% (n={m['count']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
