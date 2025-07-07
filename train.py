import argparse
import logging
import os
import random

import numpy as np
import torch
import wandb
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS, BiVisionTransformer, \
    SiVisionTransformer, MultiViT
from models.modeling_resnet import BiResNet, UniResNet, UniVGGNet
from utils.data_utils import get_loader
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)
wandb.login()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_f1_score(preds, labels):
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1_score


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset.startswith('bi_vit_'):
        model = BiVisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_class)

    elif args.dataset.startswith('si_'):
        model = SiVisionTransformer(config, args.res_type, args.img_size, zero_head=True,
                                    num_classes=args.num_class)

    elif args.dataset.startswith('multi_'):
        model = MultiViT(
            config, args.res_type, num_classes1=args.num_class, num_classes2=args.aux_class,
            img_size=args.img_size, zero_head=True
        )

    elif args.dataset.startswith('bi_res_'):
        model = BiResNet(num_classes=args.num_class)

    elif args.dataset.startswith('uni_res_'):
        model = UniResNet(num_classes=args.num_class)

    elif args.dataset.startswith('uni_vgg_'):
        model = UniVGGNet(num_classes=args.num_class)

    else:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_class)

    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid(args, model, val_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(val_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            if args.dataset.startswith('multi_'):
                x1, x2, y1, y2 = batch
                eval_loss, logits1, logits2 = model(x1, x2, y1, y2)
                y, logits = y2, logits2

            elif args.dataset.startswith('bi_') or args.dataset.startswith('si_'):
                x1, x2, y = batch
                eval_loss, logits = model(x1, x2, y)

            else:
                x, y = batch
                eval_loss, logits = model(x, y)

            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())

        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]

    precision, recall, f1_score = get_f1_score(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Precision: %2.5f" % precision)
    logger.info("Valid Recall: %2.5f" % recall)
    logger.info("Valid F1 Score: %2.5f" % f1_score)

    wandb.log({
        'val/loss': eval_losses.avg,
        'val/precision': precision,
        'val/recall': recall,
        'val/f1_score': f1_score
    })

    return f1_score


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_name,
            config={
                'learning_rate': args.learning_rate,
                'num_steps': args.num_steps,
            }
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, val_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(0)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_f1 = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)

            if args.dataset.startswith('multi_'):
                x1, x2, y1, y2 = batch
                loss, _, _ = model(x1, x2, y1, y2)

            elif args.dataset.startswith('bi_') or args.dataset.startswith('si_'):
                x1, x2, y = batch
                loss, _ = model(x1, x2, y)

            else:
                x, y = batch
                loss, _ = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if args.local_rank in [-1, 0]:
                    wandb.log({
                        'train/loss': losses.val,
                        'train/lr': scheduler.get_lr()[0]
                    })

                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    f1_score = valid(args, model, val_loader, global_step)

                    if best_f1 < f1_score:
                        save_model(args, model)
                        best_f1 = f1_score

                    model.train()

                if global_step % t_total == 0:
                    break

        losses.reset()

        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        wandb.finish()

    logger.info("Best F1 Score: \t%f" % best_f1)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", required=True,
                        help="Which downstream task.")
    parser.add_argument("--num_class", type=int, required=True,
                        help="Number of classes")
    parser.add_argument("--aux_class", type=int, default=0,
                        help="Number of auxiliary classes")
    parser.add_argument("--res_type", required=True,
                        help="Which residual to use.")

    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output",
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--wandb_name", required=True,
                        help="Name of this wandb project.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s" % (args.local_rank, args.device, args.n_gpu))

    # Set seed
    set_seed(0)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
