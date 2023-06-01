import os
import sys
from typing import List

import json
import fire
import argparse
import torch
import transformers
from datasets import load_dataset
from prompt import prompt_dict
import inspect
import copy

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

# Catch when user should re-install transformers library
assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"  # noqa: E501

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "",
        output_dir: str = "",
        # training hyperparams
        batch_size: int = 64,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 0.0003,
        cutoff_len: int = 2048,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_style: str = 'vicuna',  # options: alpaca | oig | vicuna
        save_steps=200,
        local_rank: int = 0):
    # base_model=args.base_model
    # data_path=args.data_path
    # output_dir=args.output_dir
    # batch_size=args.batch_size
    # micro_batch_size=args.micro_batch_size
    # num_epochs=args.num_epochs
    # learning_rate=args.learning_rate
    # cutoff_len=args.cutoff_len
    # lora_r=args.lora_r
    # lora_alpha=args.lora_alpha
    # lora_dropout=args.lora_dropout
    # lora_target_modules=args.lora_target_modules
    # train_on_inputs=args.train_on_inputs
    # group_by_length=args.group_by_length
    # wandb_project=args.wandb_project
    # wandb_run_name=args.wandb_run_name
    # wandb_watch=args.wandb_watch
    # wandb_log_model=args.wandb_log_model
    # resume_from_checkpoint=args.resume_from_checkpoint
    # prompt_style=args.prompt_style
    # save_steps=args.save_steps
    print(
        f"Training chatAlpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"prompt_style: {prompt_style}\n"
        f"save_steps: {save_steps}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        ### 保存运行参数
        sig = inspect.signature(train)
        param_names = [p.name for p in sig.parameters.values()]
        param_value = locals()
        # Create a dictionary with the parameter names and values
        params = {name: param_value[name] for name in param_names}

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save the dictionary as a JSON file
        with open(output_dir + "/params.json", "w") as f:
            json.dump(params, f)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    generate_prompt = prompt_dict[prompt_style]

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        with open(output_dir + "/full_prompt.txt", "a") as f:
            f.write(full_prompt)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            if prompt_style == 'alpaca':
                user_prompt = generate_prompt({**data_point, "output": ""})
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                tokenized_full_prompt["labels"] = [
                                                      -100
                                                  ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                        user_prompt_len:
                                                                        ]  # could be sped up, probably

            elif prompt_style == 'oig':
                user_prompt = generate_prompt({**data_point, "answer": ""})
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                tokenized_full_prompt["labels"] = [
                                                      -100
                                                  ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                        user_prompt_len:
                                                                        ]  # could be sped up, probably

            elif prompt_style == 'vicuna':
                system = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                roles = ("Human", "Assistant")
                BEGIN_SIGNAL = "### "
                END_SIGNAL = "\n"
                role_list = ['system']
                role_len = [len(tokenize(system, add_eos_token=False)['input_ids'])]

                for ele in data_point['conversations']:
                    if ele['from'].lower() == 'human':
                        role = roles[0]
                    elif ele['from'].lower() == 'gpt':
                        role = roles[1]
                    else:
                        role = 'unknown'

                    role_list.append(role)
                    role_len.append(len(
                        tokenize(f"{BEGIN_SIGNAL}{role}: {ele['value']}{END_SIGNAL}", add_eos_token=False)[
                            'input_ids']))
                cur_idx = 0
                break_flag = 0
                for tokenized_len, speaker in zip(role_len, role_list):
                    if speaker == "Assistant":
                        if cur_idx + tokenized_len > cutoff_len:
                            tokenized_full_prompt["labels"][cutoff_len - 1] = tokenizer.eos_token_id
                            tokenized_full_prompt["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
                            break_flag = 1
                        else:
                            tokenized_full_prompt["labels"].insert(cur_idx + tokenized_len, tokenizer.eos_token_id)
                            tokenized_full_prompt["attention_mask"].insert(cur_idx + tokenized_len, 1)
                            tokenized_full_prompt["input_ids"].insert(cur_idx + tokenized_len, tokenizer.eos_token_id)
                            cur_idx += 1
                    if speaker != "Assistant":
                        if cur_idx + tokenized_len > cutoff_len:
                            tokenized_full_prompt["labels"][cur_idx:cutoff_len] = [-100] * (cutoff_len - cur_idx)
                            break_flag = 1
                        else:
                            tokenized_full_prompt["labels"][cur_idx:cur_idx + tokenized_len] = [-100] * tokenized_len
                    if break_flag == 1:
                        break
                    cur_idx += tokenized_len

        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # data = load_dataset("json", data_files=data_path)
    data_dict = {'train': data_path + '-train.json', 'test': data_path + '-dev.json'}
    data = load_dataset("json", data_files=data_dict, num_proc=16)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = (
        data["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        data["test"].shuffle().map(generate_and_tokenize_prompt)
    )

    if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        print("#" * 20 + "Toknize example1" + "#" * 20)
        print(train_data[10])
        print(tokenizer.decode(train_data[10]["input_ids"]))
        print(train_data[10]["attention_mask"])
        label = copy.deepcopy(train_data[10]["labels"])
        label_id = [0 if x == -100 else x for x in label]
        print(tokenizer.decode(label_id))
        print(len(train_data[10]["attention_mask"]))
        print(len(train_data[10]["input_ids"]))
        print(len(train_data[10]["labels"]))
        print("#" * 20 + "Toknize example2" + "#" * 20)
        print(train_data[30])
        print(tokenizer.decode(train_data[30]["input_ids"]))
        label = copy.deepcopy(train_data[30]["labels"])
        label_id = [0 if x == -100 else x for x in label]
        print(tokenizer.decode(label_id))
        print(len(train_data[30]["attention_mask"]))
        print(len(train_data[30]["input_ids"]))
        print(len(train_data[30]["labels"]))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if int(os.environ.get("LOCAL_RANK") or 0) == 0:
        model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, default="data/convai2")
    # parser.add_argument("--output_dir", type=str, default="output")
    # parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium")
    # parser.add_argument("--micro_batch_size", type=int, default=3)
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--cutoff_len", type=int, default=2048)
    # parser.add_argument("--num_epochs", type=int, default=1)
    # parser.add_argument("--learning_rate", type=float, default=3e-4)
    # parser.add_argument("--lora_alpha", type=int, default=16)
    # parser.add_argument("--lora_dropout", type=float, default=0.05)
    # parser.add_argument("--lora_r", type=int, default=1)
    # parser.add_argument("--lora_target_modules", nargs='+', type=str, default=["q_proj"])
    # parser.add_argument('--train_on_inputs', type=str, default="True")
    # parser.add_argument('--group_by_length', type=str, default="False")
    # parser.add_argument('--wandb_project', type=str, default="")
    # parser.add_argument('--wandb_run_name', type=str, default="")
    # parser.add_argument('--wandb_watch', type=str, default="")
    # parser.add_argument('--wandb_log_model', type=str, default="")
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--resume_from_checkpoint', type=str, default="")
    # parser.add_argument("--prompt_style", type=str, default="vicuna")
    # parser.add_argument("--save_steps", type=int, default=400)
    # args = parser.parse_args()
    # train(args)
