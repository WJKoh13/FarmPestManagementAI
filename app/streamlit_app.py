from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` adds app/ to sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import ui_theme
from app.cnn_model import describe_runs
from app.conversation import NEW_CHAT_TITLE, Conversation, load_all, store_image
from app.pest_assistant import PestAssistant

WELCOME = (
    "Send a clear photo of the pest and I will tell you what it most likely is, "
    "and how to deal with it organically. You can add the crop or what you are seeing."
)
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
SUGGESTIONS = [
    "What should I check for on my tomato leaves?",
    "Is neem oil safe for bees?",
    "How do I stop cutworms from cutting seedlings?",
]

st.set_page_config(page_title="Organic Farm Pest Assistant", page_icon="🌿", layout="wide")


# --------------------------------------------------------------------- state
def new_chat() -> None:
    conversation = Conversation()
    conversation.add("assistant", WELCOME)
    st.session_state.chats.insert(0, conversation)
    st.session_state.active = conversation.id


def active_chat() -> Conversation:
    for conversation in st.session_state.chats:
        if conversation.id == st.session_state.active:
            return conversation
    return st.session_state.chats[0]


if "assistant" not in st.session_state:
    st.session_state.assistant = PestAssistant()
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "chats" not in st.session_state:
    # Conversations outlive the process now, so a farmer who closes the laptop
    # still has the identification and the advice when they come back.
    st.session_state.chats = load_all()
    if not st.session_state.chats:
        st.session_state.chats = []
        new_chat()
    st.session_state.active = st.session_state.chats[0].id
if "pending" not in st.session_state:
    st.session_state.pending = None

ui_theme.inject_css()
assistant: PestAssistant = st.session_state.assistant
chat = active_chat()


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="app-head"><h1>🌿 Pest Assistant</h1></div>', unsafe_allow_html=True)

    if st.button("＋  New chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown('<div class="side-label">Chats</div>', unsafe_allow_html=True)
    for conversation in st.session_state.chats:
        marker = "●  " if conversation.id == chat.id else ""
        label = f"{marker}{conversation.title}  ·  {ui_theme.relative_time(conversation.updated_at)}"
        if st.button(label, key=f"chat_{conversation.id}", use_container_width=True):
            st.session_state.active = conversation.id
            st.rerun()

    if len(st.session_state.chats) > 1 and st.button("Delete this chat", key="delete_chat",
                                                     use_container_width=True):
        chat.delete()
        st.session_state.chats = [c for c in st.session_state.chats if c.id != chat.id]
        st.session_state.active = st.session_state.chats[0].id
        st.rerun()

    st.divider()
    llm_ready = assistant.llm.available()
    st.markdown('<div class="side-label">Local language model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-note{"" if llm_ready else " bad"}">{assistant.llm.status_line}</div>',
        unsafe_allow_html=True,
    )
    if not llm_ready:
        st.markdown(
            '<div class="side-note">Answers come straight from the written guides. '
            "For conversational replies run <code>ollama serve</code> and "
            "<code>ollama pull phi3</code>.</div>",
            unsafe_allow_html=True,
        )

    # Farmers never need the checkpoint picker, but "why isn't my run showing
    # up?" is the question this panel exists to answer, so it stays one click away.
    with st.expander("Pest recognition model"):
        runs = describe_runs(num_classes=len(assistant.class_names))
        if not runs:
            st.markdown(
                '<div class="side-note">No runs in runs/. Import one with '
                "scripts/import_propestnet_run.py.</div>",
                unsafe_allow_html=True,
            )
        else:
            usable = [run for run in runs if run["usable"]]
            current = str(assistant.loaded.path or "")
            options = [str(run["path"]) for run in usable]
            if usable:
                chosen = st.selectbox(
                    "Checkpoint", options,
                    index=options.index(current) if current in options else 0,
                    format_func=lambda path: next(
                        f"{run['label']} — {run['model']}" + (" ⚠" if run["under_trained"] else "")
                        for run in usable if str(run["path"]) == path
                    ),
                    label_visibility="collapsed",
                )
                # Rebuilding is a few seconds of weight loading, so only do it when
                # the choice actually changed.
                if chosen != current:
                    st.session_state.assistant = PestAssistant(model_path=chosen)
                    st.rerun()

            # Runs the app cannot serve are listed with the reason rather than
            # hidden, because "my checkpoint isn't showing up" is the question this
            # panel exists to answer.
            for run in runs:
                if not run["usable"]:
                    st.markdown(
                        f'<div class="side-note bad">✗ {run["label"]} — {run["problem"]}</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()
    dark = st.toggle("Dark mode", value=st.session_state.theme == "dark")
    if dark != (st.session_state.theme == "dark"):
        st.session_state.theme = "dark" if dark else "light"
        st.rerun()


# ---------------------------------------------------------------------- page
st.markdown(
    '<div class="app-head"><h1>Organic Farm Pest Management</h1></div>'
    '<div class="app-sub">Identify the pest from a photo, then work out what to do about it '
    "— organically, and offline.</div>",
    unsafe_allow_html=True,
)

# One pill, driven by what the loader actually found, so the UI can never claim
# a healthy model while serving an under-trained or missing one.
if status := assistant.status_message:
    ui_theme.status_pill(status)

for message in chat.messages:
    with st.chat_message(message.role):
        if message.image_path and Path(message.image_path).exists():
            st.image(message.image_path, caption=message.image_name or "Uploaded photo", width=260)
        # Same order as when the turn was first drawn: what it is, how sure, what to do.
        if message.heading:
            st.markdown(message.heading)
        if message.candidates:
            ui_theme.confidence_bars(message.candidates, assistant.display_names)
        if message.content:
            st.markdown(message.content)

# The welcome card only exists while the chat is untouched; once a farmer has
# asked something, the screen belongs to the conversation.
if chat.is_empty and not st.session_state.pending:
    st.markdown('<div class="suggest-label">Or ask about</div>', unsafe_allow_html=True)
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"suggest_{index}"):
            st.session_state.pending = {"text": suggestion, "image": None, "name": None}
            st.rerun()


# --------------------------------------------------------------------- input
chat_input = st.chat_input("Describe the crop or the damage, or attach a photo",
                           accept_file=True, file_type=IMAGE_TYPES)
if chat_input:
    uploaded = chat_input.files[0] if chat_input.files else None
    st.session_state.pending = {
        "text": chat_input.text.strip(),
        # Stored beside the conversation rather than in a temporary directory,
        # which would leave a reloaded chat showing a broken image.
        "image": str(store_image(uploaded.getvalue(), uploaded.name)) if uploaded else None,
        "name": uploaded.name if uploaded else None,
    }
    st.rerun()


if pending := st.session_state.pending:
    st.session_state.pending = None
    text = pending["text"] or ("Please identify this pest." if pending["image"] else "")
    if text or pending["image"]:
        chat.add("user", text, image_path=pending["image"], image_name=pending["name"])
        chat.retitle_from(text)

        with st.chat_message("user"):
            if pending["image"]:
                st.image(pending["image"], caption=pending["name"], width=260)
            st.markdown(text)

        with st.chat_message("assistant"):
            with st.status("Reading the photo…" if pending["image"] else "Thinking…",
                           expanded=False) as status_box:
                turn = assistant.prepare_turn(text, image_path=pending["image"],
                                              conversation=chat)
                status_box.update(label="Writing your guidance…")

            if turn["heading_plain"]:
                st.markdown(turn["heading_plain"])
            if turn["candidates"]:
                ui_theme.confidence_bars(turn["candidates"], assistant.display_names)

            # Stream when the local model answers; fall back to the written guide
            # the moment it does not, which is also the no-Ollama path. An empty
            # stream is the signal — no string matching on the reply.
            # The note belongs under the bars it refers to, so it leads the body
            # rather than the heading. `fallback_body` already carries it.
            if turn["note"]:
                st.markdown(turn["note"])

            body = ""
            if turn["messages"]:
                body = str(st.write_stream(assistant.llm.stream_chat(turn["messages"])) or "")
            if not body.strip():
                body = turn["fallback_body"]
                st.markdown(body)
            elif turn["note"]:
                body = f"{turn['note']}\n\n{body}"

        chat.add("assistant", body, heading=turn["heading_plain"],
                 candidates=turn["candidates"])
        chat.save()
        st.rerun()
