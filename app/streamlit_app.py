from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# `streamlit run app/streamlit_app.py` adds app/ to sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.cnn_model import describe_runs
from app.pest_assistant import PestAssistant


WELCOME_MESSAGE = (
    "Hello! Upload a clear photo of the pest and I will identify it and suggest "
    "organic treatment options. You can also add the crop type or symptoms."
)
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]


def new_chat() -> None:
    st.session_state.chats.append(
        {"title": "New pest consultation", "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}]}
    )
    st.session_state.active_chat = len(st.session_state.chats) - 1


def save_upload(uploaded_file) -> Path:
    """Save a Streamlit upload in the operating system's temporary directory."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".jpg") as file:
        file.write(uploaded_file.getvalue())
        return Path(file.name)


st.set_page_config(page_title="Organic Farm Pest Assistant", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background: #f5f7f2; color: #1e2d20; }
        [data-testid="stSidebar"] { background: #19382a; }
        [data-testid="stSidebar"] * { color: #eef5ed !important; }
        [data-testid="stSidebar"] button { background: transparent; border: 1px solid #5e8068; }
        [data-testid="stSidebar"] button:hover { background: #2e5940; border-color: #a9cda8; }
        .block-container { max-width: 980px; padding-top: 3.25rem; padding-bottom: 7rem; }
        h1 { color: #1d462c; font-size: 2rem !important; letter-spacing: -0.035em; }
        [data-testid="stCaptionContainer"] { color: #638067; }
        [data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #e1e9df; border-radius: 16px; padding: 0.75rem 1rem; margin-bottom: 0.7rem; box-shadow: 0 3px 14px rgba(30, 65, 37, 0.04); }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #e4f0e2; border-color: #c9ddc7; }
        [data-testid="stChatInput"] { border: 1px solid #c6d8c4; border-radius: 14px; background: #ffffff; box-shadow: 0 8px 24px rgba(25, 56, 42, 0.1); }
        [data-testid="stChatInput"]:focus-within { border-color: #4f7d58; box-shadow: 0 0 0 3px rgba(79, 125, 88, 0.15); }
        [data-testid="stChatInput"] button { color: #2f6c42; }
        .model-status { margin: 1rem 0 1.5rem; padding: 0.7rem 0.9rem; background: #fbf3df; border-left: 3px solid #bd8a2d; border-radius: 6px; color: #644a1c; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "assistant" not in st.session_state:
    st.session_state.assistant = PestAssistant()
if "chats" not in st.session_state:
    st.session_state.chats = [{"title": "New pest consultation", "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}]}]
if "active_chat" not in st.session_state:
    st.session_state.active_chat = 0

chat = st.session_state.chats[st.session_state.active_chat]

with st.sidebar:
    st.title("Pest Assistant")
    if st.button("+ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("MODEL")
    runs = describe_runs(num_classes=len(st.session_state.assistant.class_names))
    if not runs:
        st.caption("No runs in runs/. Import one with scripts/import_propestnet_run.py.")
    else:
        usable = [run for run in runs if run["usable"]]
        current = str(st.session_state.assistant.loaded.path or "")
        options = [str(run["path"]) for run in usable]
        index = options.index(current) if current in options else 0

        if usable:
            chosen = st.selectbox(
                "Checkpoint", options, index=index,
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
                st.caption(f"✗ {run['label']} — {run['problem']}")

    st.divider()
    st.caption("CHATS")
    for index, saved_chat in enumerate(st.session_state.chats):
        if st.button(saved_chat["title"], key=f"chat_{index}", use_container_width=True):
            st.session_state.active_chat = index
            st.rerun()

st.title("Organic Farm Pest Management AI")
st.caption("Offline guidance for quick, organic pest-management decisions")

# One banner, driven by what the loader actually found, so the UI can never
# claim a healthy model while serving an under-trained or missing one.
status = st.session_state.assistant.status_message
if status:
    st.markdown(f'<div class="model-status">{status}</div>', unsafe_allow_html=True)
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        if image_path := message.get("image_path"):
            st.image(image_path, caption=message.get("image_name", "Uploaded pest photo"), width=280)
        st.markdown(message["content"])

chat_input = st.chat_input("Describe the crop or symptoms", accept_file=True, file_type=IMAGE_TYPES)
if chat_input:
    prompt = chat_input.text.strip()
    uploaded_file = chat_input.files[0] if chat_input.files else None
    image_path = save_upload(uploaded_file) if uploaded_file else None
    user_text = prompt.strip() or "Please identify this pest."
    user_message = {"role": "user", "content": user_text}
    if image_path:
        user_message.update({"image_path": str(image_path), "image_name": uploaded_file.name})
    chat["messages"].append(user_message)
    if chat["title"] == "New pest consultation":
        chat["title"] = user_text[:32] + ("..." if len(user_text) > 32 else "")

    with st.spinner("Analyzing your pest photo..." if image_path else "Thinking..."):
        result = st.session_state.assistant.chat_reply(user_text, image_path=image_path)
    # The reply already carries the ranked candidates and their confidences,
    # under farmer-facing names. Appending a "Detected pest: <slug>" line on top
    # restated it in raw slugs, and contradicted the reply outright whenever the
    # model was too unsure to name one.
    chat["messages"].append({"role": "assistant", "content": result["message"]})
    st.rerun()
