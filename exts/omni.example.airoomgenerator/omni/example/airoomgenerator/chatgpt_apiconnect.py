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
import openai
from .prompts import system_input, user_input, assistant_input
from .deep_search import query_items
from .item_generator import place_greyboxes, place_deepsearch_results
from openai import AsyncOpenAI


async def chatGPT_call(prompt: str):
    # Load your API key from an environment variable or secret management service
    settings = carb.settings.get_settings()
    
    apikey = settings.get_as_string("/persistent/exts/omni.example.airoomgenerator/APIKey")
    my_prompt = prompt.replace("\n", " ")

    # Create an OpenAI client
    client = AsyncOpenAI(api_key=apikey)

    # Define the role of each message in the conversation
    try:
        # 调用 OpenAI API，向模型发送问题，增加超时限制
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_input,
                },
                {
                    "role": 'user',
                    "content": user_input,
                },
                {
                    "role": 'assistant',
                    "content": assistant_input,
                },
                {
                    "role": 'user',
                    "content": my_prompt,
                }
            ],
            model="gpt-4o",
        )

        # 获取并返回生成的文本
        text = str(chat_completion.choices[0].message.content)

    except openai.APIConnectionError as e:
        carb.log_error("The server could not be reached")
        carb.log_error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        text = None
    except openai.RateLimitError as e:
        carb.log_error("A 429 status code was received; we should back off a bit.")
        text = None
    except openai.APIStatusError as e:
        carb.log_error("Another non-200-range status code was received")
        carb.log_error(e.status_code)
        carb.log_error(e.response)
        text = None
    except Exception as e:
        carb.log_error( f"发生错误: {e}")
        text = None

    if isinstance(text, str):
        return True, text
    else:
        return None, text

async def call_Generate(prim_info, prompt, use_chatgpt, use_deepsearch, response_label, progress_widget):
    run_loop = asyncio.get_event_loop()
    progress_widget.show_bar(True)
    task = run_loop.create_task(progress_widget.play_anim_forever())
    response = ""
    #chain the prompt
    area_name = prim_info.area_name.split("/World/Layout/")
    concat_prompt = area_name[-1].replace("_", " ") + ", " + prim_info.length + "x" + prim_info.width + ", origin at (0.0, 0.0, 0.0), generate a list of appropriate items in the correct places. " + prompt
    root_prim_path = "/World/Layout/GPT/"
    if prim_info.area_name != "":
        root_prim_path= prim_info.area_name + "/items/"
    
    if use_chatgpt:          #when calling the API
        objects, response = await chatGPT_call(concat_prompt)
    else:                       #when testing and you want to skip the API call
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
