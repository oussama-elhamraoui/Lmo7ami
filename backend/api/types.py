from typing import TypedDict


class ChatHistory(TypedDict):
    sender: int | None
    content: str