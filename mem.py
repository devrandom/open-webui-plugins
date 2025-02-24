"""
title: Long Term Memory Filter
author: devrandom
date: 2024-08-23
version: 1.0
license: MIT
description: A filter that processes user messages and stores them as long term memory by utilizing the mem0 framework together with qdrant and ollama
requirements: pydantic, ollama, mem0ai.  Based on a plugin written by Anton Nilsson.
"""
import re
from typing import List, Optional
from pydantic import BaseModel
import json
from mem0 import Memory
import threading


ENDS_WITH_PERIOD_RE = re.compile(r"[^.]\.$")


class Filter:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

        mem_zero_user: str = (
            "user"  # Memories belongs to this user, only used by mem0 for internal organization of memories
        )

        # Default values for the mem0 vector store
        vector_store_qdrant_name: str = "memories"
        vector_store_qdrant_url: str = "localhost"
        vector_store_qdrant_port: int = 6333
        vector_store_qdrant_dims: int = (
            3072  # Need to match the vector dimensions of the embedder model
        )
        
        # Provider selection
        llm_provider: str = "openai"  # Can be "openai" or "ollama"
        embedder_provider: str = "openai"  # Can be "openai" or "ollama"

        # Default values for the mem0 language model
        ollama_llm_model: str = "llama3.1:latest"  # This model need to exist in ollama
        ollama_llm_temperature: float = 0
        ollama_llm_tokens: int = 8000
        ollama_llm_url: str = "http://host.docker.internal:11434"

        openai_llm_model: str = "gpt-4o"
        openai_llm_temperature: float = 0.2
        openai_llm_tokens: int = 1500

        # Default values for the mem0 embedding model
        ollama_embedder_model: str = (
            "nomic-embed-text:latest"  # This model need to exist in ollama
        )
        ollama_embedder_url: str = "http://host.docker.internal:11434"

        openai_embedder_model: str = "text-embedding-3-large"

    def __init__(self):
        self.type = "filter"
        self.name = "Memory Filter"
        self.user_messages = []
        self.thread = None
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )
        self.m = self.init_mem_zero()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")

        user = self.valves.mem_zero_user

        if isinstance(body, str):
            body = json.loads(body)

        all_messages = body["messages"]

        if True:
            message_text = ""
            for m in all_messages:
                if m["role"] == "user":
                    message = m["content"].strip()
                    if not ENDS_WITH_PERIOD_RE.search(message):
                        message = message + "."
                    message_text += message + " "

            if self.thread and self.thread.is_alive():
                print("Waiting for previous memory to be done")
                self.thread.join()

            self.thread = threading.Thread(
                target=self.m.add, kwargs={"messages": message_text, "user_id": user}
            )

            print("Text to be processed in to a memory:")
            print(message_text)

            self.thread.start()
            self.user_messages.clear()

        last_message = all_messages[-1]["content"]
        print("searching for: " + last_message)
        memories = self.m.search(last_message, user_id=user)

        if memories:
            fetched_memory = memories[0]["memory"]
        else:
            fetched_memory = ""

        print("Memory added to the context:")
        print(fetched_memory)

        if fetched_memory:
            all_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Here is context retrieved from previous conversation: "
                    + str(fetched_memory),
                },
            )

        print("Final body to send to the LLM:")
        print(body)

        return body

    def init_mem_zero(self):
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.valves.vector_store_qdrant_name,
                    "host": self.valves.vector_store_qdrant_url,
                    "port": self.valves.vector_store_qdrant_port,
                    "embedding_model_dims": self.valves.vector_store_qdrant_dims,
                },
            },
        }
        
        # Configure LLM based on provider selection
        if self.valves.llm_provider == "ollama":
            config["llm"] = {
                "provider": "ollama",
                "config": {
                    "model": self.valves.ollama_llm_model,
                    "temperature": self.valves.ollama_llm_temperature,
                    "max_tokens": self.valves.ollama_llm_tokens,
                    "url": self.valves.ollama_llm_url,
                },
            }
        elif self.valves.llm_provider == "openai":
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "model": self.valves.openai_llm_model,
                    "temperature": self.valves.openai_llm_temperature,
                    "max_tokens": self.valves.openai_llm_tokens,
                },
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.valves.llm_provider}. Must be 'openai' or 'ollama'.")
        
        # Configure embedder based on provider selection
        if self.valves.embedder_provider == "ollama":
            config["embedder"] = {
                "provider": "ollama",
                "config": {
                    "model": self.valves.ollama_embedder_model,
                    "url": self.valves.ollama_embedder_url,
                },
            }
        elif self.valves.embedder_provider == "openai":
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": self.valves.openai_embedder_model,
                },
            }
        else:
            raise ValueError(f"Unsupported embedder provider: {self.valves.embedder_provider}. Must be 'openai' or 'ollama'.")

        m = Memory.from_config(config)
        return m
