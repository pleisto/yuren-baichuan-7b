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

import argparse
import logging
from typing import List, Optional
import os

import gradio as gr
import torch
from PIL.Image import Image
from webui.constants import (
    APP_ROOT,
    CONCURRENT_COUNT,
    IMAGE_PLACEHOLDER,
    ChatbotValue,
    Conversation,
    Messages,
    description,
    description_top,
    small_and_beautiful_theme,
    title,
)
from webui.overwrites import postprocess, reload_javascript
from webui.utils import (
    cancel_outputing,
    delete_last_conversation,
    is_stop_word_or_prefix,
    load_tokenizer_image_processor_and_model,
    render_assistant_message,
    render_user_message,
    reset_state,
    reset_textbox,
    sample_decode,
    shared_state,
    transfer_input,
)
from yuren_core.constants import IM_END_TOKEN
from yuren_core.errors import MaxTokenLengthError

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

parser = argparse.ArgumentParser(description="Run the webui for Yuren LLM demo")
parser.add_argument("model_name_or_path", type=str, help="model name or path")
parser.add_argument("--load_8bit", type=bool, default=True, help="load 8bit model")
parser.add_argument(
    "--server_name",
    type=str,
    default=None,
    help=(
        'server_name: to make app accessible on local network, set this to "0.0.0.0". Can be set by environment'
        ' variable GRADIO_SERVER_NAME. If None, will use "127.0.0.1".'
    ),
)
parser.add_argument(
    "--share",
    type=bool,
    default=False,
    help=(
        "share: whether to create a publicly shareable link for the interface. Creates an SSH tunnel to make your UI"
        " accessible from anywhere. "
    ),
)


args = parser.parse_args()

tokenizer, model, device, image_processor = load_tokenizer_image_processor_and_model(
    args.model_name_or_path, load_8bit=args.load_8bit
)


def predict(
    text: str,
    chatbot: ChatbotValue,
    history: List[Conversation],
    top_p: float,
    temperature: float,
    max_length_tokens: int,
    image: Optional[Image] = None,
):
    if text == "":
        if image is None:
            yield chatbot, history, "请输入问题或上传图片", image
            return
        else:
            # Set default text prompt when user only upload images
            text = "请描述一下图片的内容"

    if image is not None and text.endswith(IMAGE_PLACEHOLDER) is False:
        text = text + f"\n{IMAGE_PLACEHOLDER}"

    try:
        for output in sample_decode(
            text,
            history,
            model,
            tokenizer,
            image_processor,
            device=device,
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
            image=image,
        ):
            if is_stop_word_or_prefix(output, [IM_END_TOKEN]) is False:
                output = output.strip("\n").strip()
                chatbot_conversations = []
                # update history conversations
                for conversation in history:
                    human, bot = conversation["messages"]
                    chatbot_conversations.append(
                        [
                            render_user_message(human, conversation["image"]),
                            render_assistant_message(bot),
                        ]
                    )
                # add current conversation
                chatbot_conversations.append([render_user_message(text, image), render_assistant_message(output)])

                new_history = history + [Conversation(messages=Messages(user=text, assistant=output), image=image)]
                yield chatbot_conversations, new_history, "正在生成回答...", None

            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield chatbot_conversations, new_history, "已停止生成回答", None
                    return
                except:  # noqa: E722
                    pass

    except MaxTokenLengthError:
        yield chatbot, history, "您输入的内容加上历史消息超过了最大Token长度限制，请缩短内容或者删除历史消息。", image
        return

    torch.cuda.empty_cache()

    try:
        conversation = chatbot_conversations[-1]
        logging.info(
            f"Finish generating answer: \n User: {conversation[0]} \n Assistant: {conversation[1]} \n\n",
        )
        yield chatbot_conversations, new_history, "回答完毕", None
    except:  # noqa: E722
        pass


def retry(
    chatbot: ChatbotValue,
    history: List[Conversation],
    top_p: float,
    temperature: float,
    max_length_tokens: int,
):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, "当前会话内容为空，无法重新回答。"
        return
    chatbot.pop()
    conversation = history.pop()
    for x in predict(
        conversation["messages"].user,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        conversation["image"],
    ):
        yield x


gr.Chatbot.postprocess = postprocess

with open(f"{APP_ROOT}/assets/style.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    # history is a list of 'Conversation'
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="yuren-chat")
            with gr.Row():
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Enter text", container=False)
                with gr.Column(min_width=50, scale=1):
                    submitBtn = gr.Button("✈")
                with gr.Column(min_width=50, scale=1):
                    cancelBtn = gr.Button("中止")

            with gr.Row():
                emptyBtn = gr.Button(
                    "🧹 开始新会话",
                )
                retryBtn = gr.Button("🔄 重新回答")
                delLastBtn = gr.Button("🗑️ 删除最后一轮对话")

        with gr.Column():
            imagebox = gr.Image(type="pil")

            with gr.Column():
                with gr.Accordion("Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.42,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=4096,
                        step=8,
                        interactive=True,
                        label="Max Tokens",
                    )

    gr.Markdown(description)

    predict_args = dict(
        fn=predict,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            imagebox,
        ],
        outputs=[chatbot, history, status_display, imagebox],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[chatbot, history, top_p, temperature, max_length_tokens],
        outputs=[chatbot, history, status_display, imagebox],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    # Chatbot
    cancelBtn.click(cancel_outputing, [], [status_display])
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    user_input.submit(**transfer_input_args).then(**predict_args)

    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )

demo.title = os.getenv("YUREN_WEB_TITLE", "羽人 7b")

if __name__ == "__main__":
    reload_javascript()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(share=args.share, server_name=args.server_name)
