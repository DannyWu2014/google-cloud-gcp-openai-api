#!/usr/bin/env python3

# Copyright 2023-2024 Nils Knieling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import secrets
import time
# FastAPI
from typing import List, Optional

# Google Vertex AI
import google.auth
# LangChain
import uvicorn
# Anthropic
from anthropic import AnthropicVertex
from anthropic.types import MessageParam
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.cloud import aiplatform
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Google authentication
credentials, project_id = google.auth.default()

# Get environment variable
host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", 8000))
debug = os.environ.get("DEBUG", False)
simple_debug = os.environ.get("SIMPLE_DEBUG", False)
print(f"Endpoint: http://{host}:{port}/")
# Google Cloud
project = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", project_id)
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east5")
print(f"Google Cloud project identifier: {project}")
print(f"Google Cloud location: {location}")
# LLM chat model name to use
# Token limit determines the maximum amount of text output from one prompt
default_max_output_tokens = os.environ.get("MAX_OUTPUT_TOKENS", "81920")
# Sampling temperature,
# it controls the degree of randomness in token selection
default_temperature = os.environ.get("TEMPERATURE", "0.5")
# How the model selects tokens for output, the next token is selected from
default_top_k = os.environ.get("TOP_K", "40")
# Tokens are selected from most probable to least until the sum of their
default_top_p = os.environ.get("TOP_P", "0.8")
# API key
default_api_key = f"sk-{secrets.token_hex(21)}"
api_key = os.environ.get("OPENAI_API_KEY", default_api_key)
print(f"API key: {api_key}")

app = FastAPI(
    title='OpenAI API',
    description='APIs for sampling from and fine-tuning language models',
    version='2.0.0',
    servers=[{'url': 'https://api.openai.com/'}],
    contact={
        "name": "GitHub",
        "url": "https://github.com/Cyclenerd/google-cloud-gcp-openai-api",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

aiplatform.init(
    project=project,
    location=location,
)

vertex_client = AnthropicVertex(region=location, project_id=project_id)


class Message(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


@app.get("/")
def read_root():
    max_tokens: int = 1024
    model: str = "claude-3-opus@20240229"
    messages = [
        {
            "role": "user",
            "content": "Send me a recipe for banana bread.",
        }
    ]

    # vert_response = construct_vertex_message_stream(
    #     model=model,
    #     messages=messages,
    #     max_tokens=max_tokens,
    # )
    #
    # async def stream():
    #     yield json.dumps(
    #         generate_stream_response_start(model),
    #         ensure_ascii=False
    #     )
    #     with vert_response as rsp:
    #         for chunk in rsp.text_stream:
    #             yield json.dumps(
    #                 generate_stream_response(chunk, model),
    #                 ensure_ascii=False
    #             )
    #     yield json.dumps(
    #         generate_stream_response_stop(model),
    #         ensure_ascii=False
    #     )
    #
    # return EventSourceResponse(stream(), ping=10000)

    # vert_response = construct_vertex_message(
    #     model=model,
    #     messages=messages,
    #     max_tokens=max_tokens,
    # )
    # return JSONResponse(content=generate_response(vert_response))

    return {
        "Vertex AI": aiplatform.__version__
    }


def generate_stream_response_start(model: str):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{
            "delta": {"role": "assistant"},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response(chunk, model):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": ts,
        "model": model,
        "choices": [{
            "delta": {"content": chunk},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response_stop(model: str):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "delta": {},
            "index": 0,
            "finish_reason": "stop"
        }]
    }


def generate_response(vertex_response):
    id = f"cmpl-{secrets.token_hex(12)}"
    ts = int(time.time())
    return {
        "id": id,
        "object": "chat.completion",
        "created": ts,
        "model": vertex_response.model,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": vertex_response.content[0].text},
            "finish_reason": "stop"
        }]
    }


def get_location_by(model: str):
    # @param ["claude-3-sonnet@20240229", "claude-3-haiku@20240307", "claude-3-opus@20240229"]
    # model = "claude-3-sonnet@20240229"
    if model == "claude-3-sonnet@20240229":
        available_regions = ["us-central1", "asia-southeast1"]
    elif model == "claude-3-haiku@20240307":
        available_regions = ["us-central1", "europe-west4"]
    else:
        available_regions = ["us-east5"]
    return available_regions[0]


def construct_vertex_message(model: str, messages: List[Message], max_tokens: int, temperature: float = 0.5):
    message_params: list[MessageParam] = []
    system_prompt = ""
    for message in messages:
        role = message.role
        content = message.content
        if role == "system":
            system_prompt = content
        else:
            message_params.append({
                "role": role,
                "content": content
            })
    if simple_debug:
        print(f"system_prompt = {system_prompt}")
    response = vertex_client.messages.create(
        max_tokens=max_tokens,
        messages=message_params,
        model=model,
        system=system_prompt,
        temperature=temperature
    )
    if debug:
        print("=== Response ===")
        print(type(response))
        print(response)
    return response


def construct_vertex_message_stream(model: str, messages: List[Message], max_tokens: int, temperature: float = 0.5):
    message_params: list[MessageParam] = []
    system_prompt = ""
    for message in messages:
        role = message.role
        content = message.content
        if role == "system":
            system_prompt = content
        else:
            message_params.append({
                "role": role,
                "content": content
            })
    if simple_debug:
        print(f"system_prompt = {system_prompt}")
    return vertex_client.messages.stream(
        max_tokens=max_tokens,
        messages=message_params,
        model=model,
        system=system_prompt
    )


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request):
    """
    Creates a model response for the given chat conversation.

    https://platform.openai.com/docs/api-reference/chat/create
    """

    # Authorization via OPENAI_API_KEY
    if request.headers.get("Authorization").split(" ")[1] != api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key is wrong!")

    if debug:
        print(f"body = {body}")

    # Get user question
    if not len(body.messages):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    # Overwrite defaults
    # model="claude-3-opus@20240229",
    model_name = body.model
    temperature = float(body.temperature or default_temperature)
    max_output_tokens = int(body.max_tokens or default_max_output_tokens)

    if debug:
        print(f"stream = {body.stream}")
        print(f"model = {body.model}")
        print(f"temperature = {temperature}")
        print(f"max_output_tokens = {max_output_tokens}")
        print(f"messages = {body.messages}")

    if simple_debug:
        print(f"max_output_tokens = {max_output_tokens}")

    # Wrapper around Vertex AI large language models
    if body.stream:
        vert_response = construct_vertex_message_stream(
            model=model_name,
            messages=body.messages,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        async def stream():
            yield json.dumps(
                generate_stream_response_start(model_name),
                ensure_ascii=False
            )
            with vert_response as rsp:
                for chunk in rsp.text_stream:
                    yield json.dumps(
                        generate_stream_response(chunk, model_name),
                        ensure_ascii=False
                    )
            yield json.dumps(
                generate_stream_response_stop(model_name),
                ensure_ascii=False
            )

        return EventSourceResponse(stream(), ping=10000)
    else:
        vert_response = construct_vertex_message(
            model=model_name,
            messages=body.messages,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
        return JSONResponse(content=generate_response(vert_response))


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
