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

import os
from typing import List, NamedTuple, Optional, Tuple, TypedDict, Union

import gradio as gr
from PIL.Image import Image
from yuren_core.constants import IMAGE_END_TOKEN, IMAGE_PATCH_TOKEN, IMAGE_START_TOKEN

title = os.getenv("YUREN_WEB_TITLE", "羽人7b Demo")
description_top = """\
<div align="left" style="margin:16px 0">
For Internal Use Only</div>
"""
description = """\
<div align="center" style="margin:16px 0">
&copy; 2023 Pleisto
</div>
"""
CONCURRENT_COUNT = 100

ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

small_and_beautiful_theme = gr.themes.Monochrome()

IMAGE_PLACEHOLDER = IMAGE_START_TOKEN + (IMAGE_PATCH_TOKEN * 256) + IMAGE_END_TOKEN

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


class Messages(NamedTuple):
    """
    Message is the data structure that is iterated over in the conversations
    """

    user: str
    assistant: str


class Conversation(TypedDict):
    messages: Messages
    image: Optional[Image]


ChatbotValue = List[List[Union[str, Tuple[str], Tuple[str, str], None]]]
