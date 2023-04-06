# Alpacapalooza: A Multi-Turn Chat Dataset with Alpaca

Alpacapalooza is a chat dataset that aims to help researchers develop models for instruction-following in multi-turn conversations. The dataset is an extension of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data, which contains multi-turn instructions and their corresponding actions.

In this dataset, we use ChatGPT (`GPT-3.5-turbo`) to generate follow-up utterances and continue the conversation with ChatGPT. This process results in multi-turn conversations where the simulated user provides instructions and ChatGPT responds accordingly.

**In the name Alpacapalooza, "[palooza](https://en.wiktionary.org/wiki/palooza)" means an exaggerated event.*

![many_alpacas_chatting](many_alpacas_chatting.jpg)

<center>(AI painting of many alpacas chatting)</center>

## Data

The data in this repository consists of a JSON file `alpacapalooza_data.json` containing the multi-turn instruction-following conversations. Each item in the data file represents a different conversation between a simulated user and the ChatGPT. The data currently contain a total of 10,000 conversations with 95,558 utterances.

The conversations in the dataset begin with an instruction followed by a response from ChatGPT, which are from the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data. The conversation then continues with the user providing further instructions and ChatGPT responding accordingly. 

Each item in `alpacapalooza_data.json` is a list of dictionaries, each dictionary contains the following fields:

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

To generate the Alpacapalooza dataset, we started with the original [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) data. We used the instructions from this dataset as the starting point for our conversations.

### Utterance Generation

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

## TODO

- [x] Release 10k data
- [ ] Release 20k data
- [ ] A translated Chinese version of our data
- [ ] Llama-7b-lora model
- [ ] Llama-7b fine-tuning model

## Citation

Please cite the repo if you use the data in this repo.

```
@misc{alpacapalooza,
  author = {Ning Bian and Hongyu Lin and Yaojie Lu and Xianpei Han and Le Sun and Ben He },
  title = {Alpacapalooza: A Multi-Turn Chat Dataset with Alpaca},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/casnlu/Alpacapalooza}},
}
```