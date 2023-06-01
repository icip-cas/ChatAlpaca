import logging
from typing import List, Tuple
import argparse
import gradio as gr
from prompt import prompt_dict
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer




# 清理缓存
def clear_torch_cache():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class ChatBot:
    def __init__(self, tokenizer, model, prompt_style) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_style = prompt_style

    @staticmethod
    def get_prompt_by_history(history: List[Tuple[str, str]], query: str, prompt_style):
        """ 自定义交互过程
        """
        generate_prompt = prompt_dict[prompt_style]
        lst = []
        for ele in history:
            lst.append(ele[0])
            lst.append(ele[1])
        history = lst
        instruction = history + [query, ""]
        roles = ('user', 'assistant')
        prompt = generate_prompt({'conversations': [
            {'from': roles[i % 2], 'value': instruction[i]} for i in range(len(instruction))
        ]}).strip()
        print('---'*50+'prompt'+'---'*50)
        print(prompt)
        return prompt

    def chat(
            self,
            query: str,
            history: List[Tuple[str, str]] = [],
            max_length: int = 1024,
            num_beams=4,
            # do_sample=True,
            top_p=0.75,
            temperature=0.1,
            top_k=40,
            #repetition_penalty=4.1,
            **kwargs
    ):
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            # "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "top_k": top_k,
            #"repetition_penalty": repetition_penalty,
            **kwargs,
        }
        # if not history:
        #     prompt = self.get_prompt_by_history([], query)
        # else:
        #     prompt = self.get_prompt_by_history(history, query)
        if not history:
            prompt = self.get_prompt_by_history([], query, self.prompt_style)
        else:
            prompt = self.get_prompt_by_history(history, query, self.prompt_style)
        input_ids = self.tokenizer([prompt], return_tensors="pt")
        input_ids = input_ids.to("cuda")
        outputs = self.model.generate(**input_ids, **gen_kwargs)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        response = response.strip().replace(prompt, "")
        history.append((query, response))
        return response, history

    def predict(self, input, history=[], max_turn=20):
        max_boxes = max_turn * 2
        clear_torch_cache()
        response, history = self.chat(input, history)
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value=query))
            updates.append(gr.update(visible=True, value=response))
        if len(updates) < max_boxes:
            updates = updates + \
                      [gr.Textbox.update(visible=False)] * (max_boxes - len(updates))
        logging.info("Input : " + input)
        logging.info("Output: " + response)
        logging.info("-------------------------------------------")
        return [history] + updates

    def start_service(self, host, port, max_turn=20, concurrency_count=3):
        with gr.Blocks(css="footer{display:none !important}", title="Chat Demo ") as demo:
            gr.Markdown(
                """
                # chatalpaca-20k
                finetune llama-7b on chatalpaca-20k dataset
                """
            )
            state = gr.State([])
            text_boxes = []
            for i in range(max_turn * 2):
                if i % 2 == 0:
                    label = "提问："
                else:
                    label = "回复："
                text_boxes.append(gr.Textbox(visible=False, label=label))

            with gr.Row():
                with gr.Column(scale=4):
                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(
                        container=False)
                with gr.Column(scale=1):
                    button = gr.Button("Generate")
            txt.submit(self.predict, [txt, state], [state] + text_boxes)
            button.click(self.predict, [txt, state], [state] + text_boxes)
        print("start ")
        demo.queue(concurrency_count=concurrency_count, api_open=False)
        demo.launch(share=False, server_name=host, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_name', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7868)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--prompt_style', type=str, default='vicuna_lora')
    parser.add_argument('--max_turn', type=int, default=20)
    parser.add_argument('--concurrency_count', type=int, default=3)
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.info(f"Loading tokenizer from {options.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(options.model_path)
    logging.info(f"Loading model from {options.model_path}")
    device = 'cuda'
    load_8bit: bool = False
    model = LlamaForCausalLM.from_pretrained(
        options.model_path,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    chat_bot = ChatBot(
        tokenizer=tokenizer,
        model=model,
        prompt_style=options.prompt_style
    )

    chat_bot.start_service(
        host=options.server_name,
        port=options.server_port,
        max_turn=options.max_turn,
        concurrency_count=options.concurrency_count,
    )