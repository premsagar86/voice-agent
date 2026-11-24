# Voice layer for Claude Code

Talk to Claude Code and hear the answer. Two parts:

1. **Hooks** (`speak_hook.py`): the normal `claude` session speaks for itself.
   - Claude needs permission -> speaks the request.
   - Claude shows a plan -> speaks the plan as key points (only the plan).
   - Task finished -> speaks a short summary.
   - `F10` or the spoken word "stop" silences it (`stop_hotkey`, `stop_listen`).
2. **Voice app** (`run.py`): push-to-talk (hold `F9`) or hands-free. Whisper (CPU) turns speech into text and
   types it into your real Claude Code CLI; voice commands change its permission mode.

## Install (any PC)

    git clone https://github.com/premsagar86/voice-agent voice-agent
    cd voice-agent
    python install.py --handsfree     # drop --handsfree for push-to-talk (hold F9)

`install.py` installs the packages, creates `config.json`, adds the hooks to `~/.claude/settings.json`
(using this folder's real path) and downloads the speech model. Restart the Claude CLI once afterwards.
Needs Python 3.9+ and Claude Code. Windows works out of the box; see "Install on any PC" below for macOS/Linux.

Read modes: say `full` to hear whole replies, `brief` for key points, `short summary` for 1-3 sentences.
`done_mode` in config.json is `follow` by default, so finished tasks and plans obey that spoken choice.

## Run

    .\start-voice.ps1            # console
    .\start-voice.ps1 -Tray      # tray icon: listening on/off, stop, read mode, open config, quit
    python run.py --test-mic     # check the microphone and what Whisper hears

Targets (`"target"` in config.json):
- `interactive` (default): finds the running Claude CLI process and sends keys straight to its console,
  so focus does not matter. Replies are spoken by the hooks.
- `headless`: runs a hidden `claude -p`; the app speaks the reply and can ask for permission out loud.

## Requirements

- Windows 10/11 (macOS/Linux: see "Install on any PC"), a microphone, headphones recommended.
- Python 3.10 to 3.13 (3.9 and 3.14 are untested), and Claude Code installed with `claude` on PATH.
- Allow microphone access: Windows Settings > Privacy > Microphone.
- Run the Claude CLI in a normal console window (Windows Terminal, PowerShell, VS Code terminal).
- About 500 MB for the speech model (downloaded once). CPU only is fine; `base.en` is the fastest, `small.en` more accurate.

Tested on one Windows 11 PC (VS Code terminal). Other setups are not verified yet; issues and fixes are welcome.

## Voice commands (say the whole phrase)

| Say | Does |
|---|---|
| `stop`, `enough`, `hush` (or press F10) | silence speech, interrupt Claude (Esc) |
| `repeat that` | say the last answer again |
| `brief` / `key points`, `short summary`, `explain in detail` | read mode for app and hooks |
| `new session` | `/clear` in the CLI (headless: fresh session) |
| `auto mode`, `plan mode`, `accept edits on mode`, `manual mode` (also "change to ...") | sets the real CLI mode: it reads the footer and presses Shift+Tab until it matches, then tells you what it saw |
| `what mode` | reads the mode shown on the CLI screen |
| `approve` | headless: allow the tool Claude was refused (a bare "yes" never does) |

Hands-free: `"listen_mode": "auto"` plus a wake word, e.g. `"wake_word": "hey jarvis"`.
- Say the wake word once; from then on everything you say goes to Claude until the session ends
  (`"wake_timeout": -1`; use N for "N seconds of quiet", `0` for "wake word every time").
- A new Claude session (or `/clear`, or "new session") closes the conversation: say the wake word again.
- `chat` opens the conversation without the wake word. `esc` / `escape` presses Escape in the CLI to interrupt Claude.
- `goodbye` / `that's all` closes the conversation early. `mute` ignores everything except `unmute` / `Claude unmute` (mic stays open). `"mute_mode": "off"` closes the mic completely (no mic icon) until you press `F8` (`"unmute_hotkey"`).
- `"wake_words": ["hey jarvis", "claude", "listen"]` lists several wake phrases; any one unmutes the app.
- A sound-alike of the wake word is accepted, and short clicks or noises do not trigger or extend a recording.
- `"vad_silence": 3` waits 3 s of quiet before sending, so natural pauses do not cut you off.
- `"persona": "jarvis"` makes spoken summaries calm and butler-like.
- Mode commands ("plan mode", "auto mode", ...) change the mode silently; set `"announce_mode": true` to hear a confirmation. Failures are always spoken.
- `"clap_approve": true` (hands-free only): a double clap approves what Claude is asking. Plan screen: option 2
  (Yes, manually approve edits). Tool permission: option 1 (Yes). Anything that looks destructive (rm -rf, delete,
  force push...) is never approved by clap. Raise `"clap_threshold"` (default 0.15) if a desk bump triggers it.
- Use headphones so the mic does not hear Claude.

Risky requests (delete, rm -rf, force push, drop table...) wait for a spoken `confirm`.

## Install on any PC

    git clone / copy this folder anywhere, then:
    python install.py          # packages, config.json, hooks in ~/.claude/settings.json, speech model
    python voice.py status     # start/stop: python voice.py start | stop | restart

- Windows: works out of the box (speech via built-in SAPI).
- macOS: speech via `say`. Run `claude` inside tmux so voice can type into it. Grant the terminal Accessibility/Input Monitoring for hotkeys.
- Linux: `sudo apt install espeak-ng tmux`; run `claude` inside tmux. Global hotkeys (F9/F10) need root or the `input` group; otherwise use hands-free (`"listen_mode": "auto"`) and say "stop".
- Remove the hooks and the `/voice` command again: `python install.py --uninstall`.

After installing, inside Claude Code use `/voice start`, `/voice stop`, `/voice restart`, `/voice status`.
`/voice stop` also mutes spoken replies until the next `/voice start`.

## Contributing

Issues and pull requests are welcome. Run `python -m pytest tests` before sending a change.

## License

MIT, see `LICENSE`.

## Config

`config.example.json` lists every key. `config.json` overrides it. 
## Status

| Phase | State |
|---|---|
| 1 Hooks | done |
| 2 Voice loop | done |
| 3 Hands-free, barge-in, commands | done |
| 4 Approvals and safety | done |
| 5 Polish | tray, wake word (transcript based), launcher, config example done; installer build is optional (`build.ps1`) |

## Troubleshooting
- Nothing happens when I talk: run `python run.py --test-mic`; check Windows microphone permission and the default input device.
- Install fails on `faster-whisper` / `ctranslate2`: use Python 3.10 to 3.13.
- `/voice` not found: run `python install.py` again and restart Claude Code.
- Hands-free hears itself: use headphones.
- Mode did not change: run.py's startup line names the CLI process. Set `cli_pid` if it picked the wrong one.
- "stop" not heard: use headphones (the mic hears the speaker) or press F10.
- Hard crash: see `crash.log`. Errors go to `speak_hook.log`.
- Tests: `python -m pytest tests`
