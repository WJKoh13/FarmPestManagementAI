# How the app works

Written to be read aloud. If you can explain this page, you can answer the
questions in the demo.

---

## The three parts

```
   ┌─────────────┐        ┌─────────────┐        ┌──────────────────────┐
   │ FRONT END   │  HTTP  │  BACK END   │        │      AI ENGINE       │
   │             │ ─────► │             │ ─────► │                      │
   │  Streamlit  │ ◄───── │  FastAPI    │ ◄───── │  ProPestNet (CNN)    │
   │             │  JSON  │             │        │  qwen2.5:3b (LLM)    │
   │  the screen │        │  the boss   │        │  reference library   │
   └─────────────┘        └─────────────┘        └──────────────────────┘
     port 8501              port 8000              all on this laptop
```

**Front end** (`app/streamlit_app.py`) — the screen. Buttons, the chat, the
photo. It does no thinking at all. It has no CNN and no language model in it; it
sends a message and draws the reply.

**Back end** (`app/main.py`) — the boss. Takes the message, runs the agent, sends
the answer back. It remembers nothing between requests: the front end sends the
whole conversation each time.

**AI engine** — the three things that actually know something. The CNN that
recognises pests, the small language model that talks, and the folder of
reference documents.

Everything runs on your machine. Nothing is sent to the internet.

---

## What happens when you send a photo

1. You pick a photo and type "these are all over my kale".
2. The front end uploads the photo to the back end, which saves it and returns
   where it put it.
3. The front end sends: your message, the file location, and the chat so far.
4. The back end tells the language model: *"The farmer attached a photograph,
   saved at /some/path. Their note: these are all over my kale."*
5. **The language model decides on its own to call `classify_pest_image`.**
6. The back end runs the CNN on the photo and hands back the pest name.
7. The model then calls `lookup_treatment_guide` to get the approved advice.
8. The model writes the answer using what the tools gave it.
9. The front end draws it: what it is, how sure, what to do.

**Step 5 is the important one.** Nowhere in the code does it say "if there is a
photo, run the CNN". The model is given a list of things it can do, and it picks.
That is what "function calling, not IF-ELSE" means.

---

## The three tools

The model can ask for exactly three things. It cannot do anything else.

| Tool | What it does | Who wrote the answer |
|---|---|---|
| `classify_pest_image` | Runs the CNN on a photo | The CNN |
| `lookup_treatment_guide` | Fetches the approved advice for a pest | Us, hand-checked |
| `search_knowledge_base` | Searches the reference library | Us, background only |

**Why two knowledge tools and not one?** The treatment guides are short,
hand-checked, and are the *only* place a product or a dose may come from. The
reference library is long prose that explains *why* things work. If we merged
them, a paragraph about neem could become the source of a dose, and nobody
checked that paragraph as carefully. So they stay apart, and the code says so.

---

## RAG — the reference library

RAG means: before answering, go and look something up.

1. `docs/knowledge/` holds markdown files about organic pest control.
2. `scripts/build_knowledge_index.py` cuts them into 40 passages and turns each
   into a list of 768 numbers, using a local embedding model.
3. When you ask a question, the question becomes numbers too.
4. We find the passages whose numbers point in the most similar direction.
5. Those passages go to the model, which writes an answer from them.

That similarity step is one line of maths — a dot product. There is no vector
database because 40 passages do not need one.

**If the embedding model is missing** it falls back to plain word matching. Worse,
but it still works. The app never simply loses a feature.

---

## Short-term memory

Two things are remembered inside one conversation:

- **The messages.** The last 8 turns get sent to the model each time.
- **The pest.** Once a photo is identified, we remember what it was, so "how
  often do I spray **it**?" works with no second photo.

The back end stores neither. The front end owns the conversation and sends it
along with every request, which is why restarting the server loses nothing.

---

## When things go wrong

The app never shows a blank screen. If the language model is not running, or
fails, or produces nothing, the back end falls back to a fixed path that reads
the guide and answers from it directly. The reply says `"fallback": true` so we
can tell which one ran.

---

## The safety rule

The CNN is often unsure. If it is below 35% confident, the app must **not** name
a pest — it lists the possibilities and gives advice safe for all of them.

Getting this right took two attempts. The first version still leaked: the model
would see "closest matches: Wireworm..." and go look up wireworm, turning a 14%
guess into confident wireworm instructions. Now any pest the CNN was unsure about
is refused by the guide tool, and the refusal survives into later turns.

`tests/test_agent.py` pins this so it cannot come back.

---

## Running it

```bash
# once
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# every time
python scripts/run_app.py     # starts both, opens on :8501
```

Or separately, if you want to see the two processes:

```bash
uvicorn app.main:app --port 8000
streamlit run app/streamlit_app.py
```

To watch the agent think, without a browser:

```bash
python scripts/try_agent.py --image sample_images/aphids__IP025000243.jpg "on my kale"
```

---

## Likely questions

**"Is that really function calling, or just a wrapper?"** The model returns a
`tool_calls` field in Ollama's own format; we parse it and run what it asked for.
Show the tool trace in the UI, or run `try_agent.py`.

**"You force the guide lookup — isn't that cheating?"** If the model tries to
answer with no guide, we fetch one. It still chooses whether to look at a photo,
which guide it wants, and whether to search the library. The forced step is
marked "added by the app" in the trace, on screen, on purpose.

**"Why 3B and not something bigger?"** The brief says under 3B is fine for a
prototype. qwen2.5:3b is the smallest we found that calls tools reliably.

**"Why no MCP?"** The brief says function calling *or* MCP. We did function
calling. MCP would move the tools into a separate process and change nothing the
model sees — the model never speaks MCP, only the app does.
