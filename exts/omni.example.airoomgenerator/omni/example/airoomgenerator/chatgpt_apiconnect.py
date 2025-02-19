# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import carb
import aiohttp
import asyncio
import requests
from .prompts import system_input, user_input, assistant_input
from .deep_search import query_items
from .item_generator import place_greyboxes, place_deepsearch_results
from openai import AsyncOpenAI


async def chatGPT_call(prompt: str):
    # Load your API key from an environment variable or secret management service
    settings = carb.settings.get_settings()

    apikey = settings.get_as_string("/persistent/exts/omni.example.airoomgenerator/APIKey")
    my_prompt = prompt.replace("\n", " ")

    # set request info
    openai_url = "https://api.openai.com/v1/engines/gpt-4o/completions"
    openai_headers = {
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json"
    }
    open_ai_json = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_input},
            {"role": "user", "content": my_prompt}
        ]
    }

    response = requests.post(openai_url, headers=openai_headers, json=open_ai_json)

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析 JSON 响应
        response_json = response.json()
        # 提取生成的文本
        model_reply = response_json["choices"][0]["message"]["content"]
        return True, model_reply
    else:
        return None, None


async def call_Generate(prim_info, prompt, use_chatgpt, use_deepsearch, response_label, progress_widget):
    run_loop = asyncio.get_event_loop()
    progress_widget.show_bar(True)
    task = run_loop.create_task(progress_widget.play_anim_forever())
    response = ""
    # chain the prompt
    area_name = prim_info.area_name.split("/World/Layout/")
    concat_prompt = area_name[-1].replace("_",
                                          " ") + ", " + prim_info.length + "x" + prim_info.width + ", origin at (0.0, 0.0, 0.0), generate a list of appropriate items in the correct places. " + prompt
    root_prim_path = "/World/Layout/GPT/"
    if prim_info.area_name != "":
        root_prim_path = prim_info.area_name + "/items/"

    if use_chatgpt:  # when calling the API
        objects, response = await chatGPT_call(concat_prompt)
    else:  # when testing and you want to skip the API call
        data = json.loads(assistant_input)
        objects = data['area_objects_list']
    if objects is None:
        response_label.text = response
        return

    if use_deepsearch:
        settings = carb.settings.get_settings()
        nucleus_path = settings.get_as_string("/persistent/exts/omni.example.airoomgenerator/deepsearch_nucleus_path")
        filter_path = settings.get_as_string("/persistent/exts/omni.example.airoomgenerator/filter_path")
        filter_paths = filter_path.split(',')

        queries = list()
        for item in objects:
            queries.append(item['object_name'])

        query_result = await query_items(queries=queries, url=nucleus_path, paths=filter_paths)
        if query_result is not None:
            place_deepsearch_results(
                gpt_results=objects,
                query_result=query_result,
                root_prim_path=root_prim_path)
        else:
            place_greyboxes(
                gpt_results=objects,
                root_prim_path=root_prim_path)
    else:
        place_greyboxes(
            gpt_results=objects,
            root_prim_path=root_prim_path)

    task.cancel()
    await asyncio.sleep(1)
    response_label.text = response
    progress_widget.show_bar(False)
