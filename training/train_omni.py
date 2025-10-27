import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5OmniThinkerForConditionalGeneration
from transformers import SchedulerType, get_scheduler
from datasets import RLDSDataset, SpeechRLDSBatchTransform
from qwen_omni_utils import process_mm_info
import math
import numpy as np
from tqdm import tqdm
import wandb
import re
import argparse
from torch.utils.data import Subset


logger = get_logger(__name__)


def parse_pair(s: str) -> Tuple[int, int]:
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two integers like '224,224' or '224 224'")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError("Both elements must be integers")



def get_args():
    p = argparse.ArgumentParser(description="TrainingConfig CLI")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_warmup_steps", type=int, default=1000)
    p.add_argument("--max_train_steps", type=int, default=100000)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir. If omitted, defaults to output_dir.")
    p.add_argument("--load_model_weights", type=str, default=None,
                   help="Path to pretrained weights for finetuning.")
    p.add_argument("--data_root_dir", type=str, default="")
    p.add_argument("--data_mix", type=str, default="droid")
    p.add_argument("--resize_resolution", type=parse_pair, default="224,224",
                   help="e.g. '224,224' or '224 224'")
    p.add_argument("--shuffle_buffer_size", type=int, default=128)
    p.add_argument("--wandb_project_name", type=str, default="Nora VLA")
    p.add_argument("--checkpoint_save_frequency", type=int, default=8000)
    p.add_argument("--logging_frequency", type=int, default=100)
    p.add_argument("--gradient_clipping", type=float, default=None)
    p.add_argument("--future_action_window_size", type=int, default=5)
    p.add_argument("--no_cuda", action="store_true", help="Example boolean flag.")

    return p.parse_args()




# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 1,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 100000,
        output_dir: str = '',
        resume_from_checkpoint: Optional[str] = None,
        load_model_weights: Optional[str] = None,
        data_root_dir: str = "",
        data_mix: str = "droid", ## For this, please check out the data mix in /training/datasets/rlds/oxe/mixtures.py
        resize_resolution: tuple[int, int] = (224, 224),
        shuffle_buffer_size: int = 16_000, #256_000,
        wandb_project_name: str = "RoboOmni",
        checkpoint_save_frequency: int = 8000,
        logging_frequency: int = 100,
        gradient_clipping: Optional[float] = None, # Add gradient clipping option
        future_action_window_size: int=5,
        max_input_tokens: int = 16384,
        length_scan_cache: Optional[str] = None,
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        
        if resume_from_checkpoint is not None:
            self.resume_from_checkpoint = resume_from_checkpoint
        else:
            self.resume_from_checkpoint = output_dir
            
        # self.resume_from_checkpoint = output_dir ## This is used to continue a training by loadinng the optimizer states, model weights etc ... 
        self.load_model_weights = load_model_weights ## This is the path to a pretrained model weights if you want to finetune the model.
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.resize_resolution = resize_resolution
        self.shuffle_buffer_size = shuffle_buffer_size
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping
        self.future_action_window_size = future_action_window_size
        
        self.max_input_tokens = max_input_tokens
        self.length_scan_cache = length_scan_cache

        os.makedirs(output_dir, exist_ok=True)


class StepTracker:
    def __init__(self):
        self.completed_steps = 0
        self.total_loss_since_ckpt = 0.0
    def state_dict(self):
        return {
            "completed_steps": self.completed_steps,
            "total_loss_since_ckpt": self.total_loss_since_ckpt,
        }
    def load_state_dict(self, state):
        self.completed_steps = int(state.get("completed_steps", 0))
        self.total_loss_since_ckpt = float(state.get("total_loss_since_ckpt", 0.0))


import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import soundfile as sf
from tqdm import tqdm
import pickle
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset



# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig, processor: AutoProcessor, is_train: bool = True) -> RLDSDataset:
    """Loads and prepares the RLDS dataset."""
    print("[DEBUG] Entered load_and_prepare_dataset")
    base_dataset = RLDSDataset(
        data_root_dir=Path(config.data_root_dir),
        data_mix=config.data_mix,
        batch_transform=SpeechRLDSBatchTransform(),
        resize_resolution=config.resize_resolution,
        shuffle_buffer_size=config.shuffle_buffer_size if is_train else None,
        train=is_train,
        future_action_window_size=config.future_action_window_size
    )
    
    
    return base_dataset
    
    

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    AUDIO_ROOT_DIR = ""
    pixel_values = example['image']
    action = example['action']
    lang = example['lang']
    first_image = pixel_values
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])

    conv_str = example['conversation'].decode('utf-8')
    
    speech_conv = example['speech_conv'].decode('utf-8')
    speech_conv = speech_conv.split("[UNK]")
    
    conv_turns = conv_str.split("[UNK]")
    robo_convs = []
    # print("[INFO] CONV", conv_str,conv_turns)
    for turn in conv_turns:
        parts = turn.split("[ASSISTANT]")
        assistant_text = parts[1]
        robo_convs.append(assistant_text)
    
    
    user = example["user"]
    response = example["response"].decode("utf-8")

    

    messages=[]
    
    if len(response) > 0:


        # Find position of current_q + current_a in conversation
        idx = 0
        found = False
        
        while idx < len(robo_convs):
            # print("[DEBUG]", idx, robo_convs[idx], response)
            if robo_convs[idx] == response:
                found = True
                break
            idx += 1

        if not found:
            raise ValueError(f"Could not find instruction-response pair in conversation: {user}, {response}")

        # Add all conversation history up to but not including current step
        for i in range(0, idx+1):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": first_image},
                    {"type": "audio", "audio": os.path.join(AUDIO_ROOT_DIR, speech_conv[i].lstrip("./"))},
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": robo_convs[i]}],
            })

    else:
        for idx in range(0, len(robo_convs)):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": first_image},
                    {"type": "audio", "audio": os.path.join(AUDIO_ROOT_DIR, speech_conv[idx].lstrip("./"))},
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": robo_convs[idx]}],
            })
        
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": pixel_values}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": vlm_action}],
        })

    return messages

def collate_fn(examples,processor,fast_tokenizer):
        
        messages = [process_example(example,fast_tokenizer) for example in examples]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=False)
        batch_input = processor(
            text=text,
            audio=audios,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False
        )
        action_token_min = 151665
        action_token_max = 153712
        labels = batch_input['input_ids'].clone()


        assistant_start_tokens = torch.tensor([151644, 77091, 198], dtype=torch.long)
        assistant_end_token = 151645

        
        for i in range(labels.size(0)):
            seq = labels[i]
            found = False
            for j in range(len(seq) - len(assistant_start_tokens)-3, -1, -1):
                if torch.equal(seq[j:j+len(assistant_start_tokens)], assistant_start_tokens):
                    start_idx = j + len(assistant_start_tokens)

                    end_candidates = (seq[start_idx:] == assistant_end_token).nonzero(as_tuple=False)
                    if end_candidates.numel() > 0:
                        end_idx = start_idx + end_candidates[0].item()
                    else:
                        end_idx = len(seq)

                    mask = torch.ones_like(seq, dtype=torch.bool)
                    mask[start_idx:] = False
                    seq[mask] = -100
                    found = True
                    break

            if not found:
                print('[DEBUG]', text,labels)
                seq[:] = -100


        
        labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
        batch_input['labels'] = labels
        
        
        return batch_input


# --- 3. Model Initialization ---
def load_model_and_processor(config: TrainingConfig, accelerator: Accelerator) -> tuple[Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor]:
    """Loads the model and processor."""
    processor = AutoProcessor.from_pretrained("./model/Qwen2.5-Thinker-3B-added-action-tokens")
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "./model/Qwen2.5-Thinker-3B-added-action-tokens",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    fast_tokenizer = AutoProcessor.from_pretrained(
            "./model/pi_fast", trust_remote_code=True
        )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Rank {accelerator.process_index}: Model loaded with {param_count} parameters")
    accelerator.wait_for_everyone()  # Ensure all processes loaded the model
    
    for name, param in model.named_parameters():
        if 'audio_tower' in name:
            param.requires_grad = False
            print(f"Frozen: {name}")

    if config.load_model_weights: 
        tensors = {}
        from safetensors import safe_open
        with safe_open(config.load_model_weights, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
        accelerator.print("Pretrained weights loaded.")
        


    return model, processor, fast_tokenizer

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    
    """Main training loop."""
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    accelerator.dataloader_config.dispatch_batches =  False
    logger.info(accelerator.state, main_process_only=False)

    set_seed(42, device_specific=True)

    # Initialize Weights and Biases
    if accelerator.is_main_process:
        wandb.init(entity="sinwang", project="qwen-vla", mode ="offline")

    # Load model and processor
    model, processor, fast_tokenizer  = load_model_and_processor(config, accelerator)

    
    
    # Load and prepare dataset
    with accelerator.main_process_first():
        train_dataset = load_and_prepare_dataset(config, processor, is_train=True)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor,fast_tokenizer),
        num_workers=2,
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    # Initialize learning rate scheduler
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps
    )

    tracker = StepTracker()

    accelerator.register_for_checkpointing(lr_scheduler)
    accelerator.register_for_checkpointing(tracker)

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    ckpt_path = config.resume_from_checkpoint
    found_ckpt = False

    if re.fullmatch(r"steps_\d+", os.path.basename(ckpt_path)):
        found_ckpt = os.path.exists(ckpt_path)
    elif os.path.isdir(ckpt_path):
        subdirs = [d for d in os.listdir(ckpt_path) if d.startswith("steps_")]
        if subdirs:
            latest = max(subdirs, key=lambda s: int(s.split("_")[-1]))
            ckpt_path = os.path.join(ckpt_path, latest)
            found_ckpt = True

    if found_ckpt and os.path.exists(ckpt_path):
        accelerator.load_state(ckpt_path)
        accelerator.print(f"Resumed from local checkpoint: {ckpt_path}")
    else:
        accelerator.print("No checkpoint found — starting fresh.")

    # Training loop
    # Right now we assume single node training. I did not test on multi node training.
    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # completed_steps = 0
    # total_loss = 0.0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(tracker.completed_steps)

    # while completed_steps < max_train_steps:
    while tracker.completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                # total_loss += loss.detach().float()
                tracker.total_loss_since_ckpt += loss.detach().float()
                accelerator.backward(loss)

                if config.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    # completed_steps += 1
                    tracker.completed_steps += 1

                optimizer.step()
                lr_scheduler.step()


            # Logging
            # if completed_steps % config.logging_frequency == 0:
            if tracker.completed_steps % config.logging_frequency == 0 and accelerator.is_main_process:
            
                    
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2

                total_norm = total_norm**0.5
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"Step {tracker.completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}")
                wandb.log({"train_loss": loss.item(), "learning_rate": lr}, step=tracker.completed_steps)
                

            # Checkpointing
            if tracker.completed_steps% config.checkpoint_save_frequency == 0 and tracker.completed_steps > 0:
                
                if accelerator.is_main_process:
                    accelerator.save_state(os.path.join(config.output_dir, f"steps_{tracker.completed_steps}"))
                    # summary_data = {"steps": completed_steps, "train_loss": total_loss/config.checkpoint_save_frequency}
                    summary_data = {
                        "steps": int(tracker.completed_steps),
                        "train_loss": float(tracker.total_loss_since_ckpt / config.checkpoint_save_frequency)
                    }
                    with open(os.path.join(config.output_dir, "summary.jsonl"), "a") as f:
                        f.write(json.dumps(summary_data) + "\n")
                    logger.info(f"Checkpoint saved at step {tracker.completed_steps}")
                    tracker.total_loss_since_ckpt = 0.0
                    

            
            if tracker.completed_steps >= max_train_steps:
                break


    # Save final checkpoint
    accelerator.save_state(os.path.join(config.output_dir, f"steps_{tracker.completed_steps}"))
    if accelerator.is_main_process:
        
        checkpoint_path = os.path.join(config.output_dir, f"steps_{tracker.completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")
        wandb.finish()

def main():
    args = get_args()
    # Initialize training configuration
    # config = TrainingConfig()
    config = TrainingConfig(
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_warmup_steps=args.num_warmup_steps,
        max_train_steps=args.max_train_steps,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        load_model_weights=args.load_model_weights,
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        resize_resolution=args.resize_resolution,
        shuffle_buffer_size=args.shuffle_buffer_size,
        wandb_project_name=args.wandb_project_name,
        checkpoint_save_frequency=args.checkpoint_save_frequency,
        logging_frequency=args.logging_frequency,
        gradient_clipping=args.gradient_clipping,
        max_input_tokens=args.max_input_tokens
    )

    # Set up basic logging
    logging.basicConfig(
        filename=f"{config.output_dir}/train.log",
        filemode="a", 
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # logging.getLogger().addFilter(SystemPromptFilter())

    # for handler in logging.getLogger().handlers:
    #     handler.addFilter(SystemPromptFilter())



    if dist.is_initialized():
        print(f"Current backend: {dist.get_backend()}")

    # Run the training
    train(config)

if __name__ == "__main__":
    main()