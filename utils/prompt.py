def alpaca_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def oig_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if len(data_point["question"]) == 1:
        return f"""Hello! Welcome to our chatbot. Please answer the question.

### Question:
{data_point["question"][0][7:]}

### Response:
{data_point["answer"][5:]}"""
    else:
        return f"""Hello! Welcome to our chatbot. Please read the conversation history and answer the question.

### Conversation history:
""" + '\n'.join(data_point["question"][:-1]) + f"""
### Question:
{data_point["question"][-1][7:].strip(':')}

### Response:
{data_point["answer"][5:].strip(':')}"""


def vicuna_lora_prompt(data_point):
    system = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    roles = ("Human", "Assistant")
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"

    res = system

    # sorry about the formatting disaster gotta move fast
    conversations = data_point['conversations']

    for ele in conversations:
        if ele['from'].lower() == 'user':
            role = roles[0]
        elif ele['from'].lower() == 'assistant':
            role = roles[1]
        else:
            role = 'unknown'

        res += f"{BEGIN_SIGNAL}{role}: {ele['value']}{END_SIGNAL}"

    return res

def vicuna_hf_prompt(data_point):
    system = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    roles = ("USER", "ASSISTANT")
    BEGIN_SIGNAL = ""
    END_SIGNAL = "\n"

    res = system

    # sorry about the formatting disaster gotta move fast
    conversations = data_point['conversations']

    for ele in conversations:
        if ele['from'].lower() == 'user':
            role = roles[0]
        elif ele['from'].lower() == 'assistant':
            role = roles[1]
        else:
            role = 'unknown'

        res += f"{BEGIN_SIGNAL}{role}: {ele['value']}{END_SIGNAL}"

    return res
prompt_dict = {'alpaca': alpaca_prompt,
               'oig': oig_prompt,
               'vicuna_lora': vicuna_lora_prompt,
               'vicuna_hf': vicuna_hf_prompt
               }

