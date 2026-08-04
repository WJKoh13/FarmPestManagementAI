"""The chat's memory: turns, the pest in hand, and storage on disk.

Three things live here that the app previously did not have at all.

*History* -- `to_llm_messages` builds the role/content list the model actually
sees, so a farmer can ask a follow-up question instead of restarting.

*Pest memory* -- `PestContext` holds what the last photo was identified as, so
"how often do I spray it?" resolves without re-uploading anything. It is set
from the classifier's own top-1, never from the language model.

*Persistence* -- conversations are JSON under `.chats/`, and uploaded photos are
copied in beside them. Uploads used to go to the operating system's temporary
directory, which a reloaded chat would render as a broken image.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHATS_DIR = PROJECT_ROOT / ".chats"
IMAGES_DIR = CHATS_DIR / "images"

NEW_CHAT_TITLE = "New pest consultation"

# Turns of history sent to the model. Small local models have short useful
# context and slow down markedly as the prompt grows; the pest in hand is
# carried separately by PestContext, so older turns matter less than they look.
HISTORY_TURNS = 8


@dataclass
class Message:
    role: str
    content: str
    # The identification line, kept apart from the body so a reopened chat can
    # redraw heading -> bars -> advice in the order it was first shown, and so
    # only the body is replayed to the language model.
    heading: str = ""
    image_path: str | None = None
    image_name: str | None = None
    # Top-k (slug, confidence) from the classifier, for redrawing the confidence
    # bars when a saved chat is reopened.
    candidates: list[tuple[str, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            heading=data.get("heading", ""),
            image_path=data.get("image_path"),
            image_name=data.get("image_name"),
            candidates=[(name, float(conf)) for name, conf in data.get("candidates", [])],
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class PestContext:
    """What the last photo was identified as."""

    slug: str
    display_name: str
    confidence: float
    uncertain: bool = False
    image_path: str | None = None

    def summary(self) -> str:
        """One line for the system prompt, with no numbers a farmer shouldn't see."""
        if self.uncertain:
            return (
                f"Earlier in this conversation the farmer sent a photo. It could not be "
                f"identified with confidence; the closest match was {self.display_name}. "
                "Treat any question about 'it' or 'this pest' as being about that, but do "
                "not state the name as certain."
            )
        return (
            f"Earlier in this conversation the farmer sent a photo, identified as "
            f"{self.display_name}. Treat any question about 'it', 'them' or 'this pest' "
            "as being about that pest unless they clearly name a different one."
        )


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = NEW_CHAT_TITLE
    messages: list[Message] = field(default_factory=list)
    pest: PestContext | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- editing
    def add(self, role: str, content: str, **kwargs) -> Message:
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = message.timestamp
        return message

    def retitle_from(self, text: str, limit: int = 38) -> None:
        """Name an untouched chat after its first real question."""
        if self.title != NEW_CHAT_TITLE:
            return
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            self.title = cleaned[:limit] + ("…" if len(cleaned) > limit else "")

    @property
    def is_empty(self) -> bool:
        """True while only the greeting is on screen -- drives the welcome state."""
        return not any(message.role == "user" for message in self.messages)

    # ------------------------------------------------------------ the context
    def to_llm_messages(self, system_prompt: str, history_turns: int = HISTORY_TURNS,
                        latest: str | None = None) -> list[dict[str, str]]:
        """The role/content list to send to the model.

        ``latest`` is the turn being answered right now, which the caller has
        usually not appended yet. Images are replaced by a short note: the model
        is text-only, and the classifier's reading of the photo is already in the
        assistant turn that followed it.
        """
        payload = [{"role": "system", "content": system_prompt}]
        for message in self.messages[-history_turns * 2:]:
            if message.role not in ("user", "assistant"):
                continue
            content = message.content
            if message.image_path:
                content = f"[sent a photo of the pest] {content}".strip()
            if content:
                payload.append({"role": message.role, "content": content})
        if latest:
            if payload[-1]["role"] == "user" and payload[-1]["content"].endswith(latest):
                return payload
            payload.append({"role": "user", "content": latest})
        return payload

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pest": asdict(self.pest) if self.pest else None,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        pest = data.get("pest")
        return cls(
            id=data.get("id") or uuid.uuid4().hex[:12],
            title=data.get("title", NEW_CHAT_TITLE),
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            pest=PestContext(**pest) if pest else None,
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )

    def save(self, chats_dir: Path = CHATS_DIR) -> Path:
        chats_dir.mkdir(parents=True, exist_ok=True)
        path = chats_dir / f"{self.id}.json"
        # Write-then-rename: a crash mid-write must not leave a truncated chat
        # that then fails to load on every subsequent start.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def delete(self, chats_dir: Path = CHATS_DIR) -> None:
        (chats_dir / f"{self.id}.json").unlink(missing_ok=True)


def load_all(chats_dir: Path = CHATS_DIR) -> list[Conversation]:
    """Every saved conversation, most recently updated first."""
    conversations = []
    for path in chats_dir.glob("*.json"):
        try:
            conversations.append(Conversation.from_dict(json.loads(path.read_text("utf-8"))))
        except (OSError, ValueError, TypeError):
            # One corrupt file must not stop the app from opening the others.
            continue
    return sorted(conversations, key=lambda chat: chat.updated_at, reverse=True)


def store_image(data: bytes, filename: str, images_dir: Path = IMAGES_DIR) -> Path:
    """Copy an upload somewhere it will still exist after a restart."""
    images_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".jpg"
    path = images_dir / f"{uuid.uuid4().hex[:16]}{suffix}"
    path.write_bytes(data)
    return path


def prune_orphan_images(chats_dir: Path = CHATS_DIR, images_dir: Path = IMAGES_DIR) -> int:
    """Delete stored photos no surviving conversation refers to."""
    if not images_dir.is_dir():
        return 0
    referenced = {
        Path(message.image_path).name
        for conversation in load_all(chats_dir)
        for message in conversation.messages
        if message.image_path
    }
    removed = 0
    for path in images_dir.iterdir():
        if path.is_file() and path.name not in referenced:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def copy_into_store(source: Path, images_dir: Path = IMAGES_DIR) -> Path:
    """Bring a photo already on disk (an API upload, a sample image) into the store."""
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / f"{uuid.uuid4().hex[:16]}{source.suffix.lower() or '.jpg'}"
    shutil.copyfile(source, destination)
    return destination
