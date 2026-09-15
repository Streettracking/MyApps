import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import {
  AgentRoster,
  ChatBubble,
  QuestionWidget,
  SecretRequest,
} from '../components'

describe('Grokbot core components', () => {
  it('renders roster and selects an agent', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()

    render(
      <AgentRoster
        agents={[
          {
            id: 'a',
            name: 'Alpha',
            role: 'Research',
            status: 'idle',
            initials: 'Al',
          },
          {
            id: 'b',
            name: 'Beta',
            role: 'Ops',
            status: 'working',
            initials: 'Be',
          },
        ]}
        selectedId="a"
        onSelect={onSelect}
      />,
    )

    expect(screen.getByText('Grokbot')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Beta/i }))
    expect(onSelect).toHaveBeenCalledWith('b')
  })

  it('shows SendMessage bubble content', () => {
    render(<ChatBubble>Принял задачу</ChatBubble>)
    expect(screen.getByTestId('chat-bubble')).toHaveTextContent('Принял задачу')
  })

  it('emits question widget selection', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()

    render(
      <QuestionWidget
        prompt="Как часто?"
        options={[{ id: 'weekly', label: 'Раз в неделю' }]}
        onSelect={onSelect}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Раз в неделю' }))
    expect(onSelect).toHaveBeenCalledWith('weekly')
  })

  it('submits secret without leaving value in the form', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()

    render(<SecretRequest label="API key" onSubmit={onSubmit} />)

    const input = screen.getByLabelText('API key')
    await user.type(input, 'super-secret')
    await user.click(screen.getByRole('button', { name: 'Сохранить' }))

    expect(onSubmit).toHaveBeenCalledWith('super-secret')
    expect(input).toHaveValue('')
  })
})
