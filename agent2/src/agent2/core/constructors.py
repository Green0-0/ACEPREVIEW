from __future__ import annotations

import platform
from datetime import datetime
from typing import TYPE_CHECKING

from agent2.core.objects import Constructor

if TYPE_CHECKING:
    from agent2.core.agent import Agent


class DefaultConstructor(Constructor):
    """Initialize the agent's chat history."""

    def __init__(self, key: str = "PROMPT"):
        self.key = key

    def setup(self, agent: Agent, prompt: str) -> None:
        content = ((agent.config.prompt_wrap_top or "") + agent.config.prompt_wrap_bottom).replace(
            f"{{{{{self.key}}}}}", prompt
        )
        agent.rollout.raw_chat.extend([
            {"role": "system", "content": agent.config.system_prompt},
            {"role": "user", "content": content},
        ])

    def followup(self, agent: Agent, prompt: str) -> None:
        agent.rollout.raw_chat.append({
            "role": "user",
            "content": agent.config.followup_prompt_wrapper.replace(f"{{{{{self.key}}}}}", prompt),
        })


class _ProviderConstructor(Constructor):
    def __init__(self, key: str, format_lite: str, format_full: str):
        self.key = key
        self.format_lite = format_lite
        self.format_full = format_full

    def value(self, agent: Agent) -> str:
        raise NotImplementedError

    def _apply(self, agent: Agent, messages: list[dict]) -> None:
        value = self.value(agent)
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.replace(
                    f"{{{{{self.key}}}}}", self.format_lite.format(value)
                ).replace(
                    f"{{{{{self.key}*}}}}", self.format_full.format(value)
                )

    def setup(self, agent: Agent, prompt: str) -> None:
        self._apply(agent, agent.rollout.raw_chat)

    def followup(self, agent: Agent, prompt: str) -> None:
        messages = agent.rollout.raw_chat
        if messages and messages[-1].get("role") == "user":
            self._apply(agent, messages[-1:])


class DateConstructor(_ProviderConstructor):
    """Provide the current date to the agent chat."""

    def __init__(self, key: str, format_lite: str = "{}", format_full: str = "The current date is {}."):
        super().__init__(key, format_lite, format_full)

    def value(self, agent: Agent) -> str:
        return datetime.now().isoformat()


class DirectoryConstructor(_ProviderConstructor):
    """Provide the working directory structure to the agent chat."""

    def __init__(self, key: str, format_lite: str = "{}", format_full: str = "The directory structure is:\n{}", dir_format: str = "full"):
        super().__init__(key, format_lite, format_full)
        self.dir_format = dir_format

    max_chars: int | None = None
    """Class-wide listing budget. None lists every path; otherwise directories deeper than
    the largest depth that fits are collapsed into ``dir/ (N files)`` lines."""

    def value(self, agent: Agent) -> str:
        return compact_listing(sorted(agent.workspace.paths()), self.max_chars)


def compact_listing(paths: list[str], max_chars: int | None) -> str:
    """Every path, or a depth-limited listing of at most ``max_chars`` characters."""
    full = "\n".join(paths)
    if max_chars is None or len(full) <= max_chars:
        return full
    depth = max((p.count("/") for p in paths), default=0)
    text = ""
    for limit in range(depth, -1, -1):
        lines: list[str] = []
        counts: dict[str, int] = {}
        for path in paths:
            parts = path.split("/")
            if len(parts) - 1 <= limit:
                lines.append(path)
            else:
                folder = "/".join(parts[: limit + 1]) + "/"
                if folder not in counts:
                    lines.append(folder)
                counts[folder] = counts.get(folder, 0) + 1
        text = "\n".join(
            f"{line} ({counts[line]} files; use list_folder to see them)" if line in counts else line
            for line in lines
        )
        if len(text) <= max_chars:
            return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n... (listing truncated)"


class OSInfoConstructor(_ProviderConstructor):
    """Provide operating-system information to the agent chat."""

    def __init__(self, key: str, format_lite: str = "{}", format_full: str = "The operating system is {}."):
        super().__init__(key, format_lite, format_full)

    def value(self, agent: Agent) -> str:
        return f"{platform.system()} {platform.release()}"
