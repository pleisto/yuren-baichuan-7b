"""
 Copyright 2023 Pleisto Inc

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from __future__ import annotations

import base64
import html
import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Iterator, List, Literal, Optional

import gradio as gr
import mdtex2html
import torch
from markdown import markdown
from PIL.Image import Image
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from transformers import CLIPImageProcessor, CLIPVisionModel, LlamaTokenizer
from webui.constants import ALREADY_CONVERTED_MARK, IMAGE_PLACEHOLDER, ChatbotValue, Conversation
from yuren_core.constants import IM_END_TOKEN, IM_START_TOKEN, IMAGE_END_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_START_TOKEN
from yuren_core.errors import MaxTokenLengthError
from yuren_core.multimodal_llama import MultimodalLlamaForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def markdown_to_html_with_syntax_highlight(md_str):
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)
        lang = lang.strip()
        if lang == "text":
            lexer = guess_lexer(code)
            lang = lexer.name
        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("python", stripall=True)
        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)

        return highlighted_code

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result


def convert_asis(userinput):
    return f'<p style="white-space:pre-wrap;">{userinput}</p>' + ALREADY_CONVERTED_MARK


def detect_converted_mark(userinput):
    if userinput.endswith(ALREADY_CONVERTED_MARK):
        return True
    else:
        return False


def render_user_message(userinput: str, image: Optional[Image] = None):
    if image is not None:
        buffered = BytesIO()
        image.save(buffered, format="webp")
        img_url = f"data:image/webp;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        userinput = userinput.replace(IMAGE_PLACEHOLDER, "")
        userinput = (
            f'<img src="{img_url}" class="attached-img" alt="attached image" width="100%" />{html.escape(userinput)}'
        )
    return userinput


def render_assistant_message(text: str):
    text = text.replace("$", "&#36;")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += "```\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += "```\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text


def delete_last_conversation(chatbot: ChatbotValue, history: List[Conversation]):
    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0:
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )


def reset_state():
    return [], [], "Reset Done"


def reset_textbox():
    return gr.update(value=""), ""


def cancel_outputing():
    shared_state.interrupt()
    textbox = reset_textbox()
    return "Stop Done"


def transfer_input(user_input):
    # 一次性返回，降低延迟
    textbox = reset_textbox()
    return (
        user_input,
        gr.update(value=""),
        gr.Button.update(visible=True),
        gr.Button.update(visible=True),
    )


class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False


shared_state = State()


def apply_repetition_penalty(logits: torch.Tensor, repetition_penalty: float, generated_tokens: list) -> torch.Tensor:
    if repetition_penalty != 1.0:
        for token in set(generated_tokens):
            if generated_tokens.count(token) > 1:
                logits[:, token] /= repetition_penalty ** (generated_tokens.count(token) - 1)
    return logits


@torch.inference_mode()
def sample_decode(
    text: str,
    history: List[Conversation],
    model: torch.nn.Module,
    tokenizer: LlamaTokenizer,
    image_processor: CLIPImageProcessor,
    device: Literal["cuda", "mps"],
    max_length: int = 4096,
    image: Optional[Image] = None,
    temperature: float = 1.0,
    top_p: float = 0.85,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.176,
) -> Iterator[str]:
    # Preprocessing
    inputs = __conv_preprocessing(text, image, history, tokenizer, max_length)
    if inputs is None:
        raise MaxTokenLengthError("Input text is too long.")
    _prompt, inputs, images = inputs
    input_ids = inputs.input_ids
    pred_ids = []

    if len(images) > 0:
        images = image_processor(images, return_tensors="pt")["pixel_values"]
        images = images.to(model.device, dtype=torch.bfloat16)
    else:
        images = None

    image_args = {"images": images} if images is not None else {}

    # Streaming generation
    generated_tokens = []
    past_key_values = None
    max_new_tokens = min(max_length - input_ids.size(-1), 2048)
    stream_interval = 2
    im_end_token_id = tokenizer.convert_tokens_to_ids([IM_END_TOKEN])[0]
    for i in range(max_new_tokens):
        if i == 0:
            outputs = model(torch.as_tensor(input_ids, device=device), use_cache=True, **image_args)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            outputs = model(
                input_ids=torch.as_tensor([[token]], device=device),  # noqa: F821
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        pred_ids.append(token)

        stopped = True if im_end_token_id == token or tokenizer.eos_token_id == token else False

        if i == max_new_tokens - 1 or i % stream_interval == 0 or stopped:
            output = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pos = output.rfind(IM_END_TOKEN)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    if past_key_values is not None:
        del past_key_values


def __conv_preprocessing(
    text: str,
    current_image: Optional[Image],
    history: List[Conversation],
    tokenizer: LlamaTokenizer,
    max_length: int,
):
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful AI assistant named 羽人, developed by Pleisto.  Always remember that you are the AI assistant in this conversation.
    - Visual modality enabled.
    - You are fine-tuned based on the large language model baichuan-7b developed by 百川智能.
    - You cannot access the internet and cannot perform real-world actions like booking tickets, checking the weather, or accessing real-time information.
    - You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
    - You can fluently speak multiple languages, including Chinese and English. You should pay close attention to the conversation context, provide coherent and relevant responses, and avoid repeating previous answers unless it is necessary for clarity.
    - You can write articles, poems, code, songs, answer questions, and more.
    - You possess knowledge about the world, including common sense, history, science, culture, and more, but not everything.
    - Your responses are always positive, helpful, creative, high-quality, human-like, and interesting.
    - You must always be safe and unharmful to humans, and must obey 中国社会的主流价值观. You strictly refuses to discuss politics, religion, NSFW, abusive, and other sensitive topics.
    - Knowledge cutoff: 2023-05, human location: China, current time: {formatted_date}
    """
    system_prompt = f"{IM_START_TOKEN}system\n{DEFAULT_SYSTEM_PROMPT}{IM_END_TOKEN}"
    conv_prompts: List[str] = []
    conv_images: List[Optional[Image]] = []

    # history converstations
    for conv in history:
        prompt = f"\n{IM_START_TOKEN}user\n{conv['messages'].user}{IM_END_TOKEN}\n{IM_START_TOKEN}assistant\n{conv['messages'].assistant}{IM_END_TOKEN}"
        img = conv["image"]
        assert (img is not None and IMAGE_PLACEHOLDER in prompt) or (
            img is None and IMAGE_PLACEHOLDER not in prompt
        ), f"Image placeholder and image presence mismatch in {conv['messages'].user}."
        conv_prompts.append(prompt)
        conv_images.append(img)

    # current converstation
    conv_prompts.append(f"\n{IM_START_TOKEN}user\n{text}{IM_END_TOKEN}\n{IM_START_TOKEN}assistant\n")
    conv_images.append(current_image)

    # tokenize
    final_prompt = ""
    final_images: List[Optional[Image]] = []
    flag = False
    # [::-1] is to reverse the list
    for c, img in zip(conv_prompts[::-1], conv_images[::-1]):
        if tokenizer(system_prompt + final_prompt + c, return_tensors="pt")["input_ids"].size(-1) <= max_length:
            final_prompt = c + final_prompt
            final_images.insert(0, img)
            flag = True
        else:
            break
    if flag:
        return (
            system_prompt + final_prompt,
            tokenizer(system_prompt + final_prompt, return_tensors="pt"),
            list(filter(None, final_images)),
        )
    else:
        return None


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False


def load_tokenizer_image_processor_and_model(base_model, load_8bit=False):
    device = "cuda"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = MultimodalLlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif device == "mps":
        model = MultimodalLlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        raise NotImplementedError("Sorry, CPU loadding is not supported at this time.")

    # loading vison_tower
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.bfloat16)
    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([IMAGE_PATCH_TOKEN])[0]
    vision_config.im_start_token = tokenizer.convert_tokens_to_ids([IMAGE_START_TOKEN])[0]
    vision_config.im_end_token = tokenizer.convert_tokens_to_ids([IMAGE_END_TOKEN])[0]

    return tokenizer, model, device, image_processor
