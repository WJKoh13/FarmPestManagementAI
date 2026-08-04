"""Drive the agent from a terminal, and watch it decide.

This exists to make one thing visible without a browser: which tools the model
asked for, in what order, and whether it asked or we had to. That trace is the
whole claim of the function-calling work, so it prints first and the answer
prints after it.

    python scripts/try_agent.py --image sample_images/<a photo>.jpg
    python scripts/try_agent.py "why does neem oil work?"
    python scripts/try_agent.py --image <photo> "it's on my kale, is it safe for bees?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import PestAgent  # noqa: E402
from app.conversation import Conversation, copy_into_store  # noqa: E402
from app.pest_assistant import PestAssistant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="*", help="What the farmer says.")
    parser.add_argument("--image", help="A photo to attach.")
    parser.add_argument("--follow-up", help="A second turn, to prove the chat remembers.")
    arguments = parser.parse_args()

    message = " ".join(arguments.message)
    if not message and not arguments.image:
        parser.error("give a message, an --image, or both")

    print("Loading the classifier…", flush=True)
    assistant = PestAssistant()
    print(f"  checkpoint : {assistant.loaded.path or 'none'}")
    print(f"  classes    : {len(assistant.class_names)}")
    print(f"  language   : {assistant.llm.status_line}")
    if assistant.status_message:
        print(f"  warning    : {assistant.status_message}")

    if not assistant.llm.available():
        print("\nOllama is not answering, so there is no agent to run. Start it with "
              "`ollama serve`.")
        return 1

    agent = PestAgent(assistant=assistant)
    knowledge = agent.knowledge
    print(f"  library    : {len(knowledge)} passages"
          f"{' (embedded)' if knowledge.embedded else ' (keyword search only)'}")
    chat = Conversation()

    # Uploads live beside the conversation, the same as they do in the app -- the
    # allowlist is built from the image store, so a photo outside it is refused.
    image_path = None
    if arguments.image:
        source = Path(arguments.image)
        if not source.is_file():
            print(f"\nNo such photo: {source}")
            return 1
        image_path = str(copy_into_store(source))

    def one_turn(text: str, photo: str | None) -> None:
        print("\n" + "=" * 72)
        print(f"FARMER: {text or '(photo only)'}" + (f"   [photo: {Path(photo).name}]" if photo else ""))
        print("=" * 72)

        started = time.monotonic()
        turn = agent.plan(text, image_path=photo, conversation=chat)
        thinking = time.monotonic() - started

        print("\n--- tool calls ------------------------------------------------------")
        if not turn.trace:
            print("  (none -- the model answered without calling anything)")
        for index, step in enumerate(turn.trace, start=1):
            how = "forced by the app" if step["auto"] else "the model asked"
            print(f"  {index}. {step['name']:<24} {how}")
        print(f"\n  tool phase: {thinking:.1f}s   classified: {turn.classified}   "
              f"grounded: {turn.grounded}")

        if turn.candidates:
            print("\n--- what the CNN saw ------------------------------------------------")
            for slug, confidence in turn.candidates:
                bar = "#" * int(confidence * 40)
                print(f"  {assistant.display_name_for(slug):<24} {confidence:6.1%} {bar}")

        if not turn.ok:
            print("\nThe language model failed; the app would fall back to the written guide.")
            return

        print("\n--- answer ----------------------------------------------------------")
        body = ""
        for chunk in assistant.llm.stream_chat(turn.messages):
            body += chunk
            print(chunk, end="", flush=True)
        print()

        if not body.strip():
            fallback = assistant.prepare_turn(text, image_path=photo, conversation=chat)
            print(fallback["fallback_body"])
            body = fallback["fallback_body"]

        chat.add("user", text, image_path=photo)
        chat.add("assistant", body)

    one_turn(message or "Please identify this pest.", image_path)

    if arguments.follow_up:
        # No photo this time on purpose: the pest has to come from memory.
        one_turn(arguments.follow_up, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
