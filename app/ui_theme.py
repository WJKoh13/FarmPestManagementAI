"""The chatbot's visual layer: palette, CSS, and the few custom components.

Kept out of `streamlit_app.py` so the page file reads as page logic rather than
a wall of selectors.

The old theme was hard to read for the same three reasons most dashboards are:
a cool near-white canvas under near-black text, a heavily saturated sidebar
competing with it, and warnings styled louder than the answer. This one uses a
warm paper background, softened ink, a desaturated sage sidebar, flat borders
instead of drop shadows, and generous line-height with a limited measure.

Dark mode is a second token set rather than a media query: Streamlit 1.60 has no
`[theme.dark]` config section, so the reliable approach is to pick the token set
in Python and emit it.
"""

from __future__ import annotations

import html
import time

import streamlit as st

LIGHT = {
    "bg": "#f7f6f2",             # warm paper, not cool near-white
    "surface": "#fffefb",
    "surface_alt": "#f0f1ea",
    "sidebar": "#eceee6",
    "sidebar_ink": "#33413a",
    "ink": "#2f3a30",            # softened from near-black
    "ink_soft": "#63705f",
    "ink_faint": "#8a9585",
    "line": "#e0e2d8",
    "line_strong": "#cfd4c6",
    "accent": "#4e7d5c",
    "accent_soft": "#e8f0e6",
    "accent_ink": "#2f5c3d",
    "user_bubble": "#e9efe4",
    "warn": "#8a6a24",
    "warn_bg": "#f7efdc",
    "shadow": "0 1px 2px rgba(47, 58, 48, 0.04)",
}

DARK = {
    "bg": "#181b18",
    "surface": "#20241f",
    "surface_alt": "#262b25",
    "sidebar": "#1c211d",
    "sidebar_ink": "#c6cfc2",
    "ink": "#dfe4da",            # not pure white; lower glare
    "ink_soft": "#a3ac9d",
    "ink_faint": "#77806f",
    "line": "#2f352d",
    "line_strong": "#3d453a",
    "accent": "#7fb08c",
    "accent_soft": "#26302a",
    "accent_ink": "#a9d3b3",
    "user_bubble": "#28332a",
    "warn": "#d6b271",
    "warn_bg": "#332c1d",
    "shadow": "none",
}


def tokens() -> dict[str, str]:
    return DARK if st.session_state.get("theme") == "dark" else LIGHT


def inject_css() -> None:
    """Emit the theme. Call once per rerun, before anything is drawn."""
    palette = tokens()
    # Hyphens, not the dict's underscores. `--ink_soft` emitted against a
    # `var(--ink-soft)` reference fails silently -- the declaration is simply
    # dropped and the element inherits, which is exactly how it looks: not
    # broken, just wrong.
    variables = "\n".join(
        f"        --{name.replace('_', '-')}: {value};" for name, value in palette.items()
    )
    st.markdown(
        f"""
    <style>
      :root {{
{variables}
      }}

      /* ---------------------------------------------------------- canvas */
      /* All four: the bottom bar sits outside .stApp's painted area, so in dark
         mode styling .stApp alone leaves a light strip under the chat input. */
      body, .stApp,
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"] {{ background: var(--bg); color: var(--ink); }}
      .block-container {{
        max-width: 820px;
        padding-top: 2.2rem;
        padding-bottom: 8rem;
      }}
      .stApp, .stApp p, .stApp li {{
        font-size: 1.02rem;
        line-height: 1.68;          /* the single biggest readability win */
        color: var(--ink);
      }}
      .stApp p, .stApp li {{ max-width: 72ch; }}
      .stMarkdown strong {{ color: var(--ink); font-weight: 620; }}
      a, a:visited {{ color: var(--accent-ink); }}
      code {{
        background: var(--surface-alt) !important;
        color: var(--accent-ink) !important;
        border-radius: 5px; padding: 0.08em 0.35em; font-size: 0.88em;
      }}

      /* ---------------------------------------------------------- header */
      .app-head {{
        display: flex; align-items: baseline; gap: 0.6rem;
        margin-bottom: 0.15rem;
      }}
      .app-head h1 {{
        font-size: 1.6rem !important;
        font-weight: 650;
        letter-spacing: -0.022em;
        color: var(--ink);
        margin: 0; padding: 0;
      }}
      .app-sub {{ color: var(--ink-soft); font-size: 0.92rem; margin-bottom: 1.4rem; }}

      /* Warnings are information, not an alarm: a quiet inline pill, not a
         full-width banner shouting over the answer. */
      .pill {{
        display: inline-flex; align-items: flex-start; gap: 0.45rem;
        padding: 0.42rem 0.8rem; border-radius: 10px;   /* not 999px: it wraps */
        font-size: 0.82rem; line-height: 1.45;
        background: var(--warn-bg); color: var(--warn);
        border: 1px solid color-mix(in srgb, var(--warn) 25%, transparent);
        margin-bottom: 1.1rem; max-width: 100%;
      }}
      .pill.ok {{ background: var(--accent-soft); color: var(--accent-ink); border-color: transparent; }}

      /* --------------------------------------------------------- messages */
      [data-testid="stChatMessage"] {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.85rem;
        box-shadow: var(--shadow);
        gap: 0.75rem;
      }}
      /* The farmer's own turns, tinted. Streamlit renames these test ids between
         versions -- 1.60 uses stChatMessageAvatarUser; the older
         chatAvatarIcon-user is kept so the tint does not silently vanish. */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: var(--user-bubble);
        border-color: transparent;
      }}
      [data-testid="stChatMessage"] img {{ border-radius: 10px; }}
      /* Streamlit ships bright orange and red default avatar discs, and the icon
         is a Material *glyph* rather than an svg, so `color` is what moves it.
         The disc is painted by the inner span, not the test-id element. */
      [data-testid^="stChatMessageAvatar"],
      [data-testid^="stChatMessageAvatar"] > span {{
        background: var(--accent-soft) !important;
        border: none !important;
      }}
      [data-testid^="stChatMessageAvatar"] * {{ color: var(--accent-ink) !important; }}

      /* Tables inside answers inherit the calm borders instead of Streamlit's. */
      [data-testid="stChatMessage"] table {{ border-collapse: collapse; font-size: 0.94rem; }}
      [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {{
        border-bottom: 1px solid var(--line); padding: 0.35rem 0.7rem; color: var(--ink);
      }}
      [data-testid="stChatMessage"] th {{ color: var(--ink-soft); font-weight: 560; }}

      /* ------------------------------------------------------ chat input */
      /* The visible surface is an inner div, not the test-id element itself. */
      [data-testid="stChatInput"],
      [data-testid="stChatInput"] > div {{
        background: var(--surface) !important;
        border-radius: 14px;
      }}
      [data-testid="stChatInput"] {{
        border: 1px solid var(--line-strong);
        box-shadow: var(--shadow);
      }}
      [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent);
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
      }}
      [data-testid="stChatInputTextArea"] {{ color: var(--ink) !important; }}
      [data-testid="stChatInputTextArea"]::placeholder {{ color: var(--ink-faint) !important; }}
      [data-testid="stChatInput"] button svg {{ fill: var(--accent); }}
      /* The band behind the input is painted by an unnamed div inside stBottom.
         Left alone it stays Streamlit's light default under a dark page. */
      [data-testid="stBottom"],
      [data-testid="stBottom"] > div,
      [data-testid="stBottomBlockContainer"] {{ background: var(--bg) !important; }}

      /* ---------------------------------------------------------- sidebar */
      [data-testid="stSidebar"] {{
        background: var(--sidebar);
        border-right: 1px solid var(--line);
      }}
      [data-testid="stSidebar"] * {{ color: var(--sidebar-ink); }}
      [data-testid="stSidebar"] .stButton button {{
        background: transparent;
        border: 1px solid transparent;
        color: var(--sidebar-ink);
        text-align: left;
        justify-content: flex-start;
        font-weight: 450;
        padding: 0.42rem 0.65rem;
        border-radius: 9px;
        line-height: 1.35;
      }}
      [data-testid="stSidebar"] .stButton button:hover {{
        background: color-mix(in srgb, var(--accent) 12%, transparent);
        border-color: transparent;
      }}
      [data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background: var(--accent); color: #ffffff; font-weight: 560;
        justify-content: center;
      }}
      [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
        background: var(--accent-ink); color: #ffffff;
      }}
      .side-label {{
        text-transform: uppercase; letter-spacing: 0.09em;
        font-size: 0.7rem; font-weight: 600;
        color: var(--ink-soft) !important;
        margin: 1.2rem 0 0.35rem;
      }}
      /* --ink-soft, not --ink-faint: this is prose telling the farmer how to
         start Ollama, and it has to be readable on the dark sidebar too. */
      .side-note, .side-note * {{
        font-size: 0.78rem; color: var(--ink-soft) !important; line-height: 1.55;
      }}
      .side-note.bad, .side-note.bad * {{ color: var(--warn) !important; }}
      .side-note code {{ font-size: 0.74rem; }}
      /* The folder a broken run lives in. Needed to go and fix it, but it is
         reference material rather than the message, so it sits back. */
      .side-note .side-note-path {{
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem; opacity: 0.75; word-break: break-all;
      }}
      [data-testid="stSidebar"] hr {{ border-color: var(--line); margin: 0.9rem 0; }}
      [data-testid="stSidebar"] [data-testid="stExpander"] details {{
        border: 1px solid var(--line); border-radius: 10px; background: transparent;
      }}

      /* ------------------------------------------------------- tool trace */
      .tool-trace {{
        display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 0.85rem;
      }}
      .tool-chip {{
        font-size: 0.76rem; line-height: 1.5; color: var(--ink-soft);
        background: var(--surface-alt); border: 1px solid var(--line);
        border-radius: 999px; padding: 0.12rem 0.62rem;
      }}
      /* Dashed, so a step the app added never reads as one the model chose. */
      .tool-chip.auto {{ border-style: dashed; opacity: 0.78; }}

      /* ------------------------------------------------- confidence bars */
      .bars {{ margin: 0.7rem 0 1.15rem; }}
      .bar-row {{
        display: grid; grid-template-columns: 1fr 3rem;
        align-items: center; gap: 0.75rem; margin-bottom: 0.42rem;
      }}
      .bar-name {{ font-size: 0.92rem; color: var(--ink); }}
      .bar-track {{
        height: 6px; border-radius: 999px; background: var(--surface-alt);
        overflow: hidden; margin-top: 0.24rem;
      }}
      .bar-fill {{ height: 100%; border-radius: 999px; background: var(--accent); }}
      .bar-row:not(:first-child) .bar-fill {{ opacity: 0.45; }}
      .bar-value {{
        font-size: 0.85rem; color: var(--ink-soft);
        text-align: right; font-variant-numeric: tabular-nums;
      }}

      /* ------------------------------------------------------ empty state */
      .welcome {{
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.5rem 1.6rem 1.2rem;
        margin-bottom: 1.1rem;
      }}
      .welcome h2 {{
        font-size: 1.12rem; font-weight: 600; margin: 0 0 0.4rem;
        color: var(--ink); letter-spacing: -0.01em;
      }}
      .welcome p {{ color: var(--ink-soft); margin: 0; font-size: 0.96rem; }}
      .suggest-label {{
        font-size: 0.78rem; color: var(--ink-faint); margin: 0.2rem 0 0.5rem;
        text-transform: uppercase; letter-spacing: 0.08em;
      }}
      [data-testid="stMain"] .stButton button {{
        background: var(--surface); border: 1px solid var(--line-strong);
        color: var(--ink-soft); border-radius: 999px;
        font-size: 0.88rem; font-weight: 450; padding: 0.4rem 0.9rem;
      }}
      [data-testid="stMain"] .stButton button:hover {{
        border-color: var(--accent); color: var(--accent-ink); background: var(--accent-soft);
      }}
      [data-testid="stMain"] .stButton button p {{ color: inherit; }}

      /* Streamlit's own chrome we do not need on a chat page. */
      #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}
      [data-testid="stExpander"] summary {{ color: var(--ink-soft); font-size: 0.86rem; }}
      [data-testid="stStatus"] {{
        background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
      }}
    </style>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text: str, ok: bool = False) -> None:
    icon = "✓" if ok else "!"
    st.markdown(
        f'<div class="pill{" ok" if ok else ""}"><span>{icon}</span>'
        f"<span>{html.escape(text)}</span></div>",
        unsafe_allow_html=True,
    )


def confidence_bars(candidates: list[tuple[str, float]], display_names: dict[str, str]) -> None:
    """The top-k as labelled bars.

    A farmer reads "how sure, and what else could it be" off a bar in a glance;
    the markdown table this replaces made them parse percentages. The text table
    is still what the API and the offline fallback return.
    """
    if not candidates:
        return
    rows = "".join(
        f'<div class="bar-row">'
        f'<div><div class="bar-name">{html.escape(display_names.get(name, name))}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{max(conf, 0.015) * 100:.1f}%"></div></div>'
        f"</div>"
        f'<div class="bar-value">{conf:.0%}</div>'
        f"</div>"
        for name, conf in candidates
    )
    st.markdown(f'<div class="bars">{rows}</div>', unsafe_allow_html=True)


TOOL_LABELS = {
    "classify_pest_image": "Looked at the photo",
    "lookup_treatment_guide": "Read the organic guide",
    "search_knowledge_base": "Searched the reference library",
}


def tool_trace(trace: list[dict]) -> None:
    """What the assistant did before answering, in the order it did it.

    Shown to the farmer in plain words, not tool names: "Looked at the photo"
    rather than `classify_pest_image`. The system rules forbid the assistant
    mentioning its own machinery in prose, and it would be odd for the interface
    to announce what the prose is required to hide.

    Steps the app forced rather than the model choosing are marked, because a
    trace that quietly presents both as the model's own decisions would be
    misleading in exactly the place someone would look to check.
    """
    if not trace:
        return
    chips = "".join(
        f'<span class="tool-chip{"" if not step.get("auto") else " auto"}">'
        f'{html.escape(TOOL_LABELS.get(step["name"], step["name"]))}'
        f'{"" if not step.get("auto") else " · added by the app"}</span>'
        for step in trace
    )
    st.markdown(f'<div class="tool-trace">{chips}</div>', unsafe_allow_html=True)


def relative_time(timestamp: float) -> str:
    """'just now' / '15m' / '3h' / '2d' for the chat list."""
    seconds = max(0.0, time.time() - timestamp)
    if seconds < 90:
        return "just now"
    for cutoff, divisor, suffix in ((3600, 60, "m"), (86400, 3600, "h"), (604800, 86400, "d")):
        if seconds < cutoff:
            return f"{int(seconds // divisor)}{suffix}"
    return time.strftime("%d %b", time.localtime(timestamp))
