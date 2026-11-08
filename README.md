# Days at Three Branches agent

Edit `agent.py` to give every one of your villagers a routine. The platform runs a separate `Agent` instance for each NPC, so instances do not share variables or memory. The supplied `sandbox/` directory contains the local runner, types, and village helpers. Leave it unchanged, and leave `requirements.in` and `requirements.txt` alone: the pinned packages match the server.

Start with the [Getting Started guide](https://vox-deorum.github.io/game-sandbox/students/getting-started/). Then run these commands from this folder:

```console
python -m sandbox watch  # watch your villagers and the scripted visitor
python -m sandbox test   # run the checks
python -m sandbox eval   # run repeatable automated days
```

## Files you will use

| Path | Purpose |
| --- | --- |
| `agent.py` | Your `Agent` implementation and starter TODOs. |
| `environment.md` | Rules, helpers, observations, and local commands. |
| `manifest.json` | Tells Game Sandbox where the agent class lives. |
| `season.json` | Optional local season settings downloaded from My Submissions. |
| `tests/` | Checks your submission should pass. |
| `sandbox/` | Local game, helpers, and types. Do not edit it. |
| `requirements.txt` | Exact Python package versions used by the server. |
| `requirements-dev.txt` | Test dependencies. |
| `.env.example` | Example local LLM settings. |

The starter shows `action.walk`, `action.stand`, an emote, and `use`. Read [`environment.md`](environment.md) before changing it. Begin with one behavior that you can recognize in `watch`, then make it more responsive to the people and props it sees.

Chat is optional. If you add `chat(self, inbox)` to your agent, send and receive raw message dictionaries that use canonical player IDs such as `"player_0"` and `"player_1"`. For example, return `[{"to": "player_0", "text": "Hello."}]` to send a direct message. The village helpers do not include a chat namespace. Read [`environment.md`](environment.md#chat-with-other-agents) for the inbox format, broadcasts, and delivery timing.

In `watch` and `eval`, your `Agent` controls the whole cast, and `scripted_visitor` controls the visitor. The `naive` built-in is the simple baseline for the cast. When you are ready to submit, follow the [submitting guide](https://vox-deorum.github.io/game-sandbox/students/submitting/).

## Optional LLM API

If your instructor enables model calls, follow [Using the LLM API](llm.md). Copy `.env.example` to `.env`, add the endpoint and key, and never commit either secret.

Test the connection with:

```console
python -m sandbox llm
```

## AI-Use disclosure and reflection

I used ChatGPT to update my Three Branches agent so that the agents are now capable of maintaining short basic conversations with the player while they work around the village. Built upon previous season's agents, so all of their design goals are still present. ChatGPT made the code changes; and I did not make additional manual edits. I reviewed the new agent, checked the helpers against the environment page, and tested it against the seeds before and after each change. The new agent now performs as wanted for the design goal. I approached the design of this agent with a hybrid setup. The agent has some canned responses for basic keywords - like what their job is, or what they're currently doing - and includes LLM calls for more complex questions. If an answer isn't immediately available, the agent will let the player know when they have an answer and to ask the question again. I didn't route every question through the LLM so as to avoid the use of too many tokens and to make it so the responses felt more timely.

# Design Goal

My design goal for this agent was: In the RPG game, players should notice the villagers are now able to have short conversations with the player and answer most questions about the village or their job. Players should also notice that the villagers still move around the village doing their jobs and taking breaks.
