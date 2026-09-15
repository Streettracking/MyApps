# Grokbot

Core UI components for a desktop AI employee: roster, SendMessage chat, human gates, Cursor agent cards, and the shared computer frame.

## Product objects covered

| Component | Object |
|---|---|
| `AgentRoster` | Employee / roster |
| `ChatBubble` | SendMessage voice |
| `QuestionWidget` / `SecretRequest` / `AutoReviewCard` | Human gates |
| `CursorAgentCard` | Delegated cloud work |
| `ComputerFrame` | “My computer” |

## Develop

```sh
npm install
npm run dev
npm test
npm run build
```

Demo shell: three panes — roster · chat · computer.
