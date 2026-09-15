import type { ReactNode } from 'react'

export type AgentStatus = 'idle' | 'working' | 'awaiting' | 'offline'

export type Agent = {
  id: string
  name: string
  role: string
  status: AgentStatus
  initials: string
}

const statusLabel: Record<AgentStatus, string> = {
  idle: 'Свободен',
  working: 'В работе',
  awaiting: 'Ждёт вас',
  offline: 'Офлайн',
}

type AgentRosterProps = {
  agents: Agent[]
  selectedId: string
  onSelect: (id: string) => void
}

export function AgentRoster({ agents, selectedId, onSelect }: AgentRosterProps) {
  return (
    <aside className="gb-roster" aria-label="Сотрудники">
      <div className="gb-roster__brand">
        <span className="gb-roster__mark" aria-hidden>
          G
        </span>
        <div>
          <p className="gb-roster__product">Grokbot</p>
          <p className="gb-roster__caption">Roster</p>
        </div>
      </div>

      <ul className="gb-roster__list">
        {agents.map((agent) => {
          const selected = agent.id === selectedId
          return (
            <li key={agent.id}>
              <button
                type="button"
                className={`gb-roster__row${selected ? ' is-selected' : ''}`}
                onClick={() => onSelect(agent.id)}
                aria-current={selected ? 'true' : undefined}
              >
                <span className="gb-roster__avatar" data-status={agent.status}>
                  {agent.initials}
                </span>
                <span className="gb-roster__meta">
                  <span className="gb-roster__name">{agent.name}</span>
                  <span className="gb-roster__role">{agent.role}</span>
                </span>
                <span className="gb-roster__status" data-status={agent.status}>
                  {statusLabel[agent.status]}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </aside>
  )
}

type ChatBubbleProps = {
  author?: 'agent' | 'user'
  children: ReactNode
  timestamp?: string
}

export function ChatBubble({
  author = 'agent',
  children,
  timestamp,
}: ChatBubbleProps) {
  return (
    <article
      className={`gb-bubble gb-bubble--${author}`}
      data-testid="chat-bubble"
    >
      <div className="gb-bubble__body">{children}</div>
      {timestamp ? <time className="gb-bubble__time">{timestamp}</time> : null}
    </article>
  )
}

export type QuestionOption = {
  id: string
  label: string
}

type QuestionWidgetProps = {
  prompt: string
  options: QuestionOption[]
  onSelect?: (id: string) => void
  disabled?: boolean
}

export function QuestionWidget({
  prompt,
  options,
  onSelect,
  disabled = false,
}: QuestionWidgetProps) {
  return (
    <section className="gb-widget" data-testid="question-widget" aria-label="Вопрос">
      <p className="gb-widget__prompt">{prompt}</p>
      <div className="gb-widget__options">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            className="gb-widget__option"
            disabled={disabled}
            onClick={() => onSelect?.(option.id)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p className="gb-widget__hint">Выбор завершает ход — сотрудник ждёт ответ</p>
    </section>
  )
}

type SecretRequestProps = {
  label: string
  placeholder?: string
  onSubmit?: (value: string) => void
  disabled?: boolean
}

export function SecretRequest({
  label,
  placeholder = 'Введите значение',
  onSubmit,
  disabled = false,
}: SecretRequestProps) {
  return (
    <form
      className="gb-secret"
      data-testid="secret-request"
      onSubmit={(event) => {
        event.preventDefault()
        const data = new FormData(event.currentTarget)
        const value = String(data.get('secret') ?? '')
        if (value) onSubmit?.(value)
        event.currentTarget.reset()
      }}
    >
      <label className="gb-secret__label" htmlFor="gb-secret-input">
        {label}
      </label>
      <div className="gb-secret__row">
        <input
          id="gb-secret-input"
          name="secret"
          type="password"
          className="gb-secret__input"
          placeholder={placeholder}
          autoComplete="off"
          disabled={disabled}
          required
        />
        <button type="submit" className="gb-secret__submit" disabled={disabled}>
          Сохранить
        </button>
      </div>
      <p className="gb-secret__hint">Значение уходит в хранилище секретов, не в чат</p>
    </form>
  )
}

type CursorAgentCardProps = {
  title: string
  status: 'running' | 'idle' | 'done' | 'error'
  repo: string
  href?: string
}

const agentStatusCopy: Record<CursorAgentCardProps['status'], string> = {
  running: 'Работает',
  idle: 'Ожидает',
  done: 'Готово',
  error: 'Ошибка',
}

export function CursorAgentCard({
  title,
  status,
  repo,
  href = '#',
}: CursorAgentCardProps) {
  return (
    <a
      className="gb-agent-card"
      href={href}
      data-testid="cursor-agent-card"
      data-status={status}
    >
      <div className="gb-agent-card__top">
        <span className="gb-agent-card__badge">Cursor Cloud Agent</span>
        <span className="gb-agent-card__status" data-status={status}>
          {agentStatusCopy[status]}
        </span>
      </div>
      <p className="gb-agent-card__title">{title}</p>
      <p className="gb-agent-card__repo">{repo}</p>
    </a>
  )
}

type ComputerFrameProps = {
  title?: string
  children: ReactNode
}

export function ComputerFrame({
  title = 'Мой компьютер',
  children,
}: ComputerFrameProps) {
  return (
    <section className="gb-computer" aria-label={title} data-testid="computer-frame">
      <header className="gb-computer__chrome">
        <div className="gb-computer__traffic" aria-hidden>
          <span />
          <span />
          <span />
        </div>
        <p className="gb-computer__title">{title}</p>
        <span className="gb-computer__tag">shared machine · desktop per agent</span>
      </header>
      <div className="gb-computer__screen">{children}</div>
    </section>
  )
}

type AutoReviewCardProps = {
  action: string
  reason: string
  onAllow?: () => void
  onDeny?: () => void
}

export function AutoReviewCard({
  action,
  reason,
  onAllow,
  onDeny,
}: AutoReviewCardProps) {
  return (
    <section className="gb-review" data-testid="auto-review-card" aria-label="Проверка действия">
      <p className="gb-review__eyebrow">Auto-review</p>
      <h3 className="gb-review__title">Нужно ваше решение</h3>
      <p className="gb-review__action">{action}</p>
      <p className="gb-review__reason">{reason}</p>
      <div className="gb-review__actions">
        <button type="button" className="gb-review__deny" onClick={onDeny}>
          Отклонить
        </button>
        <button type="button" className="gb-review__allow" onClick={onAllow}>
          Разрешить
        </button>
      </div>
    </section>
  )
}

type ComposerProps = {
  onSend?: (text: string) => void
  placeholder?: string
}

export function Composer({
  onSend,
  placeholder = 'Сообщение сотруднику…',
}: ComposerProps) {
  return (
    <form
      className="gb-composer"
      data-testid="composer"
      onSubmit={(event) => {
        event.preventDefault()
        const data = new FormData(event.currentTarget)
        const text = String(data.get('message') ?? '').trim()
        if (!text) return
        onSend?.(text)
        event.currentTarget.reset()
      }}
    >
      <input
        name="message"
        className="gb-composer__input"
        placeholder={placeholder}
        aria-label="Сообщение"
        autoComplete="off"
      />
      <button type="submit" className="gb-composer__send">
        Отправить
      </button>
    </form>
  )
}
