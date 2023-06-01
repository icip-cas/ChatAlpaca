# ChatAlpaca: A Multi-Turn Dialogue Corpus based on Alpaca Instructions

ChatAlpaca is a chat dataset that aims to help researchers develop models for instruction-following in multi-turn conversations. The dataset is an extension of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data, which contains multi-turn instructions and their corresponding responses.

ChatAlpaca is developed by Chinese Information Processing Laboratory at the Institute of Software, Chinese Academy of Sciences (www.icip.org.cn).

In this dataset, we use ChatGPT (`GPT-3.5-turbo`) to generate follow-up utterances and continue the conversation with ChatGPT. This process results in multi-turn conversations where the simulated user provides instructions and ChatGPT responds accordingly.

![many_alpacas_chatting](many_alpacas_chatting.jpg)

<center>(AI painting of many alpacas chatting)</center>

## Data

The data in this repository consists of a JSON file `data/chatalpaca-10k.json` containing the multi-turn instruction-following conversations. Each item in the data file represents a different conversation between a simulated user and the ChatGPT. The data currently contain a total of 10,000 conversations with 95,558 utterances.

Update: We released the `data/chatalpaca-20k.json` data and a translated Chinese version of the data (using Google Translation) `data/chatalpaca-20k-chinese-google-translation.json`. 

The conversations in the dataset begin with an instruction, followed by a response from ChatGPT, which are from the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data. The conversation then continues with the user providing further instructions and ChatGPT responding accordingly. 

Each item in `data/chatalpaca-10k.json` is a list of dictionaries, each dictionary contains the following fields:

- `role`: `str`, either "user" or "assistant", represents the role in the conversation.
- `content`: `str`, the conversation content.

Here is an example of our data.

```json
[
    {
        "role": "user",
        "content": "Identify the odd one out: Twitter, Instagram, Telegram"
    },
    {
        "role": "assistant",
        "content": "Telegram"
    },
    {
        "role": "user",
        "content": "What makes Telegram different from Twitter and Instagram?"
    },
    {
        "role": "assistant",
        "content": "Telegram is a cloud-based instant messaging app that focuses on privacy and security. Unlike Twitter and Instagram which are mainly used for following news, celebrities, and sharing images, Telegram was created as a secure messaging app for private and group communication. Telegram also offers more advanced features than Twitter and Instagram, such as the ability to schedule messages, create bots, and send encrypted messages."
    },
    {
        "role": "user",
        "content": "Can you give me an example of how the scheduling messages feature can be useful on Telegram?"
    },
    {
        "role": "assistant",
        "content": "Sure! The scheduling messages feature on Telegram can be useful in many different situations. Here's an example:\n\nLet's say you have a workgroup on Telegram, and you want to send a message to everyone in the group to remind them of a meeting. However, you know that most of the people in the group are busy at work during the time you want to send the message.\n\nThat's where the scheduling messages feature comes in. You can schedule the message to be sent at a later time when people are more likely to be free, such as during their lunch break or after work. This way, everyone in the group will receive the message at the same time, and you won't have to worry about anyone missing the important reminder.\n\nOverall, the scheduling messages feature can be really helpful for coordinating communication with groups of people on Telegram, whether it's for work, social events, or anything else."
    },
    {
        "role": "user",
        "content": "Goodbye."
    }
]
```

## Data Generation Process

To generate the ChatAlpaca dataset, we started with the original [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data. We used the instructions from this dataset as the starting point for our conversations.

### User Utterance Generation

We use ChatGPT (`GPT-3.5-turbo`) to generate follow-up utterances and continue the conversation. To do this, we provide the model with a designed prompt and the current chat history, and ask ChatGPT to generate a simulated user utterance by either following up with a further question or providing hints if the answer is wrong or if it doesn't know the answer. 

For example, given the chat history:

```
user:
Identify the odd one out: Twitter, Instagram, Telegram

assistant:
Telegram
```

the generated utterance by ChatGPT is

```
What makes Telegram different from Twitter and Instagram?
```

We use keywords to filter out utterances that sound like an AI assistant and ask ChatGPT to generate the utterances again. The filtered-out utterances include "As an AI language model, I'm here to assist you.", "Do you have any questions that I can help you with?", etc.

### Response Generation

We then use the generated user utterance as the starting point for the next turn of the conversation. The `messages` for ChatGPT in the above example is:

```json
[
    {
        "role": "user",
        "content": "Identify the odd one out: Twitter, Instagram, Telegram"
    },
    {
        "role": "assistant",
        "content": "Telegram"
    },
    {
        "role": "user",
        "content": "What makes Telegram different from Twitter and Instagram?"
    }
]
```

The response of ChatGPT is then appended to the chat history.

We continued this process until we reached a predetermined number of turns for conversation (we set to 5 turns in this repo) or the generated user utterance contains "GoodBye". The final dataset contains conversations of varying lengths, with some consisting of only a few turns and others containing up to 5 turns.

## Model Weights
We release Chatalpaca-7b-lora and Chatalpaca-7B-hf model weights, which is trained on 20k data. You can download it from [Chatalpaca-20k-hf](https://huggingface.co/eziosauditore/chatalpaca-20k-hf) and [Chatalpaca-20k-lora](https://huggingface.co/eziosauditore/chatalpaca-20k-lora). You can add these weights to the original LLaMA-7B model to acquire the full model. The corresponding commands are as followed.


### Chatalpaca-7B-hf
```bash
python utils/apply_delta.py \
    --model_name_or_path /path/to/llama-7b \
    --delta_path /path/to/delta-weights \
    --targe-model-path /path/to/output/chatalpaca-7b-hf 
```

### Chatalpaca-7B-lora
```bash
python utils/apply_lora.py \
    --model_name_or_path /path/to/llama-7b \
    --lora_path /path/to/lora-weights \
    --targe-model-path /path/to/output/chatalpaca-7b-lora 
```

## Fine-tuning
If you want to fine-tune LLaMA with your own data, please following the format of `./data/chatalpaca-20k.json` and run the fintuning script. 

### Code and Hyperparameters
Our code is based on [Fastchat](https://github.com/lm-sys/FastChat) and [alpaca-lora](https://github.com/tloen/alpaca-lora) with support of multi-turn chat data. We use similar hyperparameters as Fastchat and alpaca-lora.

#### Full Tune

| Hyperparameter   | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|------------------| ---: | ---: |-------:| ---: | ---: |
| Chatalpaca-7b-hf | 128 | 2e-5 |      6 | 2048 | 0 |

You can use the following command to train Chatalpaca-7B-hf with 8*A100(80G).
```bash
GPUS_PER_NODE=8

torchrun --nproc_per_node=8 --master_port=20001 ./train/train_hf.py \
    --model_name_or_path /path/to/llama-7b  \
    --data_path /path/to/chatalpaca-20k-data \
    --bf16 True \
    --output_dir /path/to/chatalpaca-7b-hf \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True 
```
#### Lora Tune
| Hyperparameter     | Global Batch Size | Learning rate | Epochs | Max length |           Lora_target_modules | Lora_r |
|--------------------|------------------:|--------------:|-------:| ---: |------------------------------:|-------:|
| Chatalpaca-7b-lora |                64 |          3e-4 |      3 | 2048 | [q_proj,k_proj,v_proj,o_proj] |     16 |

Use the following command to train Chatalpaca-7B-lora with single GPU, remember to split the dataset into train and eval dataset first.

```bash
python ./train/train_lora.py \
    --base_model /path/to/llama-7b  \
    --data_path /path/to/chatalpaca-please-add-train-and-eval \
    --output_dir /path/to/store/chatalpaca-7b-lora \
    --micro_batch_size 8  \
    --batch_size 64 \
    --cutoff_len 2048 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --prompt_style vicuna \
    --save_steps 200 \
    --train_on_inputs False \
    --wandb_project Chatalpaca \
    --wandb_run_name Chatalpaca-7B-lora
```

## Deploy and Inference
Using the following command to deploy the Chatalpaca-20k model to a local server.
```bash
python utils/deploy.py \
    --model_path /path/to/your-model
    --prompt_style vicuna-hf
```
After that, you can visit localhost:7860 to chat with it.

## TODO

- [x] Release 10k data
- [x] Release 20k data
- [x] A translated Chinese version of our data
- [x] LLaMA-7B-LoRA model
- [x] LLaMA-7B fine-tuning model

## Citation

Please cite the repo if you use the data in this repo.

```
@misc{ChatAlpaca,
  author = {Ning Bian and Hongyu Lin and Yaojie Lu and Xianpei Han and Le Sun and Ben He },
  title = {ChatAlpaca: A Multi-Turn Dialogue Corpus based on Alpaca Instructions},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cascip/ChatAlpaca}},
}
```

For more information please see www.icip.org.cn.