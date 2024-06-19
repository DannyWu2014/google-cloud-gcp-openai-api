#!/usr/bin/env python3


import json
import os
import secrets
import time
import datetime
import uvicorn

# FastAPI
from typing import List, Optional, Dict

from anthropic.types import MessageParam
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Anthropic
from anthropic import AnthropicVertex

# Google Vertex AI
import google.auth
from google.cloud import aiplatform

# LangChain
import langchain
from langchain_community.chat_models import ChatVertexAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Google authentication
credentials, project_id = google.auth.default()

# Get environment variable
debug = os.environ.get("DEBUG", False)
project = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", project_id)
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east5")
print(f"Google Cloud project identifier: {project}")
print(f"Google Cloud location: {location}")

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


def generate_stream_response_start():
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
        "choices": [{
            "delta": {"role": "assistant"},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response(content: str):
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
        "choices": [{
            "delta": {"content": content},
            "index": 0,
            "finish_reason": None
        }]
    }


def generate_stream_response_stop():
    ts = int(time.time())
    id = f"cmpl-{secrets.token_hex(12)}"
    return {
        "id": id,
        "created": ts,
        "object": "chat.completion.chunk",
        "model": "gpt-3.5-turbo",
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


def construct_vertex_message(temperature: float = 0.5):
    model: str = "claude-3-opus@20240229"
    system_prompt = "hi"
    max_tokens: int = 1024
    response = vertex_client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": "Send me a recipe for banana bread.",
            }
        ],
        model=model,
        system=system_prompt,
        temperature=temperature
    )
    return response


def construct_vertex_message_stream(temperature: float = 0.5):
    max_tokens: int = 1024
    model: str = "claude-3-opus@20240229"
    with vertex_client.messages.stream(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": "Send me a recipe for banana bread.",
                }
            ],
            model=model,
    ) as stream:
        # rsp = stream.text_stream
        # print(rsp)
        for text in stream.text_stream:
            print(text, end="", flush=True)


if __name__ == "__main__":
    # rsp = construct_vertex_message()
    # print(rsp.model_dump_json)
    construct_vertex_message_stream()
