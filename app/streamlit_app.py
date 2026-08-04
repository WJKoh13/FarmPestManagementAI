from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` adds app/ to sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import api_client as api
from app import ui_theme
from app.conversation import NEW_CHAT_TITLE, Conversation, PestContext, load_all

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
chat = active_chat()

# The front end holds no model and no weights. Everything it needs to describe
# the system comes from one call to the backend, refreshed each rerun so a
# backend started after the browser still shows up without a reload.
health = api.health()
backend_up = bool(health)
display_names: dict[str, str] = health.get("display_names", {})


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

    # A labelled upload control, separate from the paperclip in the composer.
    # Both work; this one is unambiguously a button, which is what a farmer
    # looks for when they have a photo and have not thought about typing yet.
    st.markdown('<div class="side-label">Upload a photo</div>', unsafe_allow_html=True)
    picked = st.file_uploader("Upload a pest photo", type=IMAGE_TYPES,
                              label_visibility="collapsed", key="sidebar_upload")
    if picked is not None and st.button("Identify this pest", use_container_width=True,
                                        type="primary"):
        stored = api.upload_image(picked.getvalue(), picked.name)
        if stored:
            st.session_state.pending = {"text": "", "image": stored, "name": picked.name}
            st.rerun()
        else:
            st.markdown('<div class="side-note bad">The backend did not accept the '
                        "photo. Is it running?</div>", unsafe_allow_html=True)

    st.divider()

    # The first thing to check when the app looks broken, so it goes first.
    st.markdown('<div class="side-label">Backend</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-note{"" if backend_up else " bad"}">'
        f'{"connected" if backend_up else "not running — start uvicorn app.main:app"}</div>',
        unsafe_allow_html=True,
    )

    llm = health.get("llm", {})
    llm_ready = bool(llm.get("ready"))
    st.markdown('<div class="side-label">Local language model</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-note{"" if llm_ready else " bad"}">'
        f'{llm.get("status_line", "unknown")}</div>',
        unsafe_allow_html=True,
    )
    if backend_up and not llm_ready:
        st.markdown(
            '<div class="side-note">Answers come straight from the written guides. '
            "For conversational replies run <code>ollama serve</code> and "
            "<code>ollama pull qwen2.5:3b</code>.</div>",
            unsafe_allow_html=True,
        )

    knowledge = health.get("knowledge", {})
    if knowledge:
        st.markdown('<div class="side-label">Reference library</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-note">{knowledge.get("passages", 0)} passages · '
            f'{"semantic search" if knowledge.get("embedded") else "keyword search"}</div>',
            unsafe_allow_html=True,
        )

    # Farmers never need the checkpoint picker, but "why isn't my run showing
    # up?" is the question this panel exists to answer, so it stays one click away.
    with st.expander("Pest recognition model"):
        catalogue = api.models()
        runs = catalogue.get("runs", [])
        if not runs:
            st.markdown(
                '<div class="side-note">No runs in runs/. Import one with '
                "scripts/import_propestnet_run.py.</div>",
                unsafe_allow_html=True,
            )
        else:
            usable = [run for run in runs if run["usable"]]
            current = catalogue.get("current") or ""
            options = [str(run["path"]) for run in usable]
            if usable:
                chosen = st.selectbox(
                    "Checkpoint", options,
                    index=options.index(current) if current in options else 0,
                    format_func=lambda path: next(
                        run["display_label"] + (" ⚠" if run["under_trained"] else "")
                        for run in usable if str(run["path"]) == path
                    ),
                    label_visibility="collapsed",
                )
                # Reloading weights takes seconds, so only ask when it changed.
                if chosen != current:
                    with st.spinner("Loading that checkpoint…"):
                        api.select_model(chosen)
                    st.rerun()

            # Runs the app cannot serve are listed with the reason rather than
            # hidden, because "my checkpoint isn't showing up" is the question this
            # panel exists to answer.
            for run in runs:
                if not run["usable"]:
                    st.markdown(
                        f'<div class="side-note bad">✗ {run["display_name"]} — {run["problem"]}<br>'
                        f'<span class="side-note-path">{run["label"]}</span></div>',
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

# One pill, driven by what the backend actually loaded, so the UI can never claim
# a healthy model while serving an under-trained or missing one.
if not backend_up:
    ui_theme.status_pill("The backend is not running. Start it with: "
                         "uvicorn app.main:app --port 8000")
elif status := health.get("classifier", {}).get("status"):
    ui_theme.status_pill(status)

for message in chat.messages:
    with st.chat_message(message.role):
        if message.image_path and Path(message.image_path).exists():
            st.image(message.image_path, caption=message.image_name or "Uploaded photo", width=260)
        # Same order as when the turn was first drawn: what it is, how sure, what to do.
        if message.heading:
            st.markdown(message.heading)
        if message.candidates:
            ui_theme.confidence_bars(message.candidates, display_names)
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
        # The backend owns the photo store: its tools read from it, and the
        # allowlist that keeps the model inside it is defined against it.
        "image": api.upload_image(uploaded.getvalue(), uploaded.name) if uploaded else None,
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
            # One call. The backend runs the language model, lets it choose its
            # tools, runs them, and returns the finished turn — this file makes no
            # decision about the pest and holds no model with which to make one.
            with st.spinner("Reading the photo…" if pending["image"] else "Thinking…"):
                reply = api.agent_turn(
                    text,
                    image_path=pending["image"],
                    history=[{"role": m.role, "content": m.content,
                              "image_path": m.image_path} for m in chat.messages[:-1]],
                    pest_name=chat.pest.slug if chat.pest else None,
                    pest_uncertain=bool(chat.pest.uncertain) if chat.pest else False,
                )

            heading_plain, candidates = "", []
            if error := reply.get("error"):
                body = error
                st.markdown(f"**{error}**\n\nStart it with "
                            "`uvicorn app.main:app --port 8000`.")
            else:
                heading_plain = reply.get("heading", "")
                note = reply.get("note", "")
                candidates = [(slug, float(score))
                              for slug, score in reply.get("candidates", [])]
                body = reply.get("answer", "")

                # What the model chose to do, shown as it happened. This is the
                # difference between claiming the model drives the CNN and showing
                # it, so it belongs on screen and not only in a log.
                ui_theme.tool_trace(reply.get("trace", []))

                # Same order as every redraw: what it is, how sure, what to do.
                if heading_plain:
                    st.markdown(heading_plain)
                if candidates:
                    ui_theme.confidence_bars(candidates, display_names)
                # The note is about the bars, so it sits under them, leading the body.
                if note:
                    st.markdown(note)
                    body = f"{note}\n\n{body}"
                st.markdown(reply.get("answer", ""))

                # Remember the pest, so the next turn resolves "it" with no photo.
                if reply.get("pest_name"):
                    slug = reply["pest_name"]
                    chat.pest = PestContext(
                        slug=slug,
                        display_name=display_names.get(slug, slug.replace("_", " ").title()),
                        confidence=float(reply.get("confidence") or 0.0),
                        uncertain=bool(reply.get("uncertain")),
                        image_path=pending["image"],
                    )

        chat.add("assistant", body, heading=heading_plain, candidates=candidates)
        chat.save()
        st.rerun()
