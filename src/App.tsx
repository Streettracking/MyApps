import { useMemo, useState } from 'react'
import {
  AgentRoster,
  AutoReviewCard,
  ChatBubble,
  Composer,
  ComputerFrame,
  CursorAgentCard,
  QuestionWidget,
  SecretRequest,
  type Agent,
} from './components'

const agents: Agent[] = [
  {
    id: 'purchasing',
    name: 'Purchasing',
    role: 'Заказы и поставщики',
    status: 'awaiting',
    initials: 'Pu',
  },
  {
    id: 'research',
    name: 'Research',
    role: 'Обзор рынка и конкурентов',
    status: 'working',
    initials: 'Re',
  },
  {
    id: 'ops',
    name: 'Ops',
    role: 'Рутины и мониторинг',
    status: 'idle',
    initials: 'Op',
  },
]

export default function App() {
  const [selectedId, setSelectedId] = useState(agents[0].id)
  const [messages, setMessages] = useState<string[]>([
    'Тяну заказы Amazon за прошлую неделю.',
  ])
  const [widgetDone, setWidgetDone] = useState(false)
  const [reviewResolved, setReviewResolved] = useState(false)

  const selected = useMemo(
    () => agents.find((agent) => agent.id === selectedId) ?? agents[0],
    [selectedId],
  )

  return (
    <div className="gb-shell">
      <AgentRoster
        agents={agents}
        selectedId={selected.id}
        onSelect={setSelectedId}
      />

      <main className="gb-chat">
        <header className="gb-chat__header">
          <p className="gb-chat__eyebrow">SendMessage · единственный голос</p>
          <h1 className="gb-chat__title">{selected.name}</h1>
          <p className="gb-chat__sub">
            Обычный текст модели остаётся внутренним монологом. В чат попадают
            только сообщения сотрудника, виджеты и карточки.
          </p>
        </header>

        <div className="gb-chat__thread">
          <ChatBubble author="user" timestamp="09:41">
            Подтяни заказы Amazon за прошлую неделю.
          </ChatBubble>

          {messages.map((text, index) => (
            <ChatBubble key={`${text}-${index}`} timestamp="09:41">
              {text}
            </ChatBubble>
          ))}

          <CursorAgentCard
            title="Parse Amazon order export"
            status="running"
            repo="Streettracking/MyApps"
          />

          {!widgetDone ? (
            <QuestionWidget
              prompt="Как часто запускать эту проверку?"
              options={[
                { id: 'daily', label: 'Каждый будний день' },
                { id: 'weekly', label: 'Раз в неделю' },
                { id: 'manual', label: 'Только по запросу' },
              ]}
              onSelect={(id) => {
                setWidgetDone(true)
                setMessages((prev) => [
                  ...prev,
                  id === 'weekly'
                    ? 'Ок — поставлю еженедельную рутину.'
                    : 'Принято, зафиксирую этот режим.',
                ])
              }}
            />
          ) : null}

          <SecretRequest
            label="Нужен API-ключ для connector’а Amazon"
            placeholder="Секрет не попадёт в транскрипт"
            onSubmit={() =>
              setMessages((prev) => [
                ...prev,
                'Ключ сохранён. Продолжаю через connector.',
              ])
            }
          />

          {!reviewResolved ? (
            <AutoReviewCard
              action="Shell: curl https://sellercentral.amazon.com/orders/export"
              reason="Действие может отправить данные во внешний сервис."
              onAllow={() => {
                setReviewResolved(true)
                setMessages((prev) => [
                  ...prev,
                  'Разрешение получено — выгружаю отчёт.',
                ])
              }}
              onDeny={() => {
                setReviewResolved(true)
                setMessages((prev) => [
                  ...prev,
                  'Ок, без этого шага. Предложу безопасный путь.',
                ])
              }}
            />
          ) : null}
        </div>

        <Composer
          onSend={(text) =>
            setMessages((prev) => [...prev, `Принял: «${text}». Начинаю.`])
          }
        />
      </main>

      <aside className="gb-side">
        <p className="gb-side__label">Два компьютера</p>
        <ComputerFrame title="Мой компьютер · Purchasing">
          <p>
            <code>/home/box/profile</code>
          </p>
          <p>memory/amazon-orders.md</p>
          <p>routines/weekly-export.json</p>
          <p>browser · sellercentral session</p>
          <p style={{ marginTop: 16, color: 'rgba(255,255,255,0.55)' }}>
            Ваш ноутбук — другой компьютер. Сюда работа не уезжает по умолчанию.
          </p>
        </ComputerFrame>
      </aside>
    </div>
  )
}
