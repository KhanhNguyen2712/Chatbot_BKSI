"""Conversation memory for chat history."""

from collections import defaultdict
from datetime import datetime
from typing import Any

from loguru import logger

from src.config import get_settings
from src.models import ChatMessage


class ConversationMemory:
    """Manage conversation history for context-aware chat."""

    def __init__(
        self,
        max_messages: int | None = None,
        window_size: int | None = None,
    ):
        settings = get_settings()
        self.max_messages = max_messages or settings.memory_max_messages
        self.window_size = window_size or 10

        # Store conversations by session_id
        self._conversations: dict[str, list[ChatMessage]] = defaultdict(list)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
        )

        self._conversations[session_id].append(message)

        # Trim if exceeds max messages
        if len(self._conversations[session_id]) > self.max_messages:
            self._conversations[session_id] = self._conversations[session_id][
                -self.max_messages :
            ]

        logger.debug(
            f"Added message to session {session_id}: {role} ({len(content)} chars)"
        )

    def get_history(
        self,
        session_id: str,
        window_size: int | None = None,
    ) -> list[ChatMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            window_size: Number of recent messages to return

        Returns:
            List of ChatMessage objects
        """
        size = window_size or self.window_size
        history = self._conversations.get(session_id, [])
        return history[-size:]

    def get_formatted_history(
        self,
        session_id: str,
        window_size: int | None = None,
    ) -> str:
        """
        Get formatted conversation history as string.

        Args:
            session_id: Session identifier
            window_size: Number of recent messages

        Returns:
            Formatted history string
        """
        history = self.get_history(session_id, window_size)

        if not history:
            return ""

        formatted = []
        for msg in history:
            role_label = "Người dùng" if msg.role == "user" else "Trợ lý"
            formatted.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted)

    def get_langchain_messages(
        self,
        session_id: str,
        window_size: int | None = None,
    ) -> list[tuple[str, str]]:
        """
        Get history in LangChain format.

        Args:
            session_id: Session identifier
            window_size: Number of recent messages

        Returns:
            List of (role, content) tuples
        """
        history = self.get_history(session_id, window_size)
        return [(msg.role, msg.content) for msg in history]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.debug(f"Cleared session: {session_id}")

    def clear_all(self) -> None:
        """Clear all conversation histories."""
        self._conversations.clear()
        logger.info("Cleared all conversation histories")

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_sessions": len(self._conversations),
            "total_messages": sum(len(msgs) for msgs in self._conversations.values()),
            "max_messages": self.max_messages,
            "window_size": self.window_size,
        }
