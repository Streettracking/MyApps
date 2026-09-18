# Design Constraints: FlyWire MB on Unitree Go2

Спецификация гибрида: **урезанный коннектом грибовидного тела Drosophila (FlyWire FAFB)** как индивидуальный контроллер выбора действия на Go2; внешняя система только задаёт условия среды и анализирует результаты.

Связанные артефакты:
- `artifacts/connectome_mb_v1.npz` — рабочий MB-подграф FlyWire FAFB-783
- `artifacts/manifest.json` — counts, источники, sha256
- `docs/fly_go2/physical_setup.md` — физические условия и оборудование
- `docs/fly_go2/CITATIONS.md` — цитирование данных
- `configs/examples/episode.schema.yaml` — поля конфига эпизода
- `configs/examples/room_v1.yaml` — пример арены
- `configs/examples/agent_local.yaml` — локальные флаги особи

---

## 1. Неизменяемые принципы

### 1.1 Individuum-first sociality

Социальное поведение **формируется только внутри особи**:
- восприятие других агентов — через **собственные** сенсоры (камера, лидар, IMU);
- ассоциации и пластичность — только в локальном графе (`W_kc_mbon` + DAN-маски из коннектома);
- решение действия (`Approach` / `Avoid` / `Freeze` / `Explore`) — только локальный MBON-decode.

Внешняя система **не является** социальным мозгом, учителем чужих наград и шиной поведения между роботами.

### 1.2 External = conditions + analysis

| Внешнее (arena PC) | Индивидуум (Go2) |
|---|---|
| Карта зон, стимулы, лимиты safety | FlyWire MB subgraph + пластичность |
| Старт/стоп эпизода, e-stop | Локальный расчёт `r` из позы и зон |
| Приём телеметрии, логи, offline-метрики | Детект conspecifics сенсорами |
| Раздача **условий** до/в начале эпизода | Выбор действия и cmd в sport/nav |

### 1.3 Connectome is real data

Стартовая проводка — из FlyWire/Codex (snapshot, например FAFB v783), не random «fly-inspired» матрица.
Допустимо сжатие subgraph; недопустимо подменять топологию произвольным MLP без ablation-контроля.

---

## 2. Границы ответственности

```
arena_pc                         go2_i
────────                         ─────
zones, stimuli, safety    →      load episode conditions
episode start/stop        →      run local loop
                          ←      telemetry (observe only)
analyze logs offline             perceive peers locally
                                 update own weights
                                 publish actions to legs
```

**Запрещено** считать социальный эффект доказанным, если он зависит от сетевых сообщений о состоянии других агентов.

---

## 3. Интерфейс конфига эпизода

Конфиг эпизода — единственный «управляющий» объект, который PC передаёт особям как **условия мира**. После `episode_start` особь работает автономно до `episode_stop` / e-stop.

### 3.1 Транспорт

- Канал: ROS2 / DDS / UDP JSON — на усмотрение реализации.
- Топик (логический): `/arena/set_config` затем `/arena/episode/start`.
- Сообщение должно содержать `config_hash` (SHA-256 канонического YAML/JSON).
- Особь **отклоняет** старт, если hash не совпал с локально принятым конфигом.

### 3.2 Поля конфига

См. `configs/examples/episode.schema.yaml`. Обязательные группы:

| Группа | Назначение |
|---|---|
| `episode` | id, mode, duration, seed |
| `connectome` | artifact + FlyWire snapshot citation |
| `arena.zones` | полигоны и значения `r` среды |
| `arena.stimuli` | метки A/B, освещение/теги |
| `agents` | id, AprilTag, стартовая поза |
| `safety` | max_speed, fence, clearance |
| `analysis` | какие метрики писать (для PC) |

`mode` влияет только на **условия и локальные флаги**, не на серверный social coaching:

- `solo` — один агент;
- `multi_independent` — несколько агентов; детект conspecifics включён;
- `multi_blind_peers` — агенты физически есть, но локальный детект conspecifics выключен (контроль);
- `ablation_random_wiring` — тот же размер сети, random graph (контроль «это не муха»).

### 3.3 Что конфиг НЕ содержит

- чужие награды / MBON / веса других агентов;
- live-подсказки Avoid/Approach;
- shared_brain / average_weights;
- список «заразить агента X событием агента Y».

---

## 4. Локальный расчёт награды `r`

### 4.1 Источник истины

После загрузки `arena.zones` особь считает `r` **сама** из своей позы и локальной копии зон.

PC **не** публикует `/arena/reward/go2_i` как обучающий сигнал.
Допустимо, чтобы PC **параллельно** считал `r` только для логов и сверки (analysis mirror), но робот не должен от этого зависеть.

### 4.2 Алгоритм (нормативный)

На каждом тике MB (`f_mb`, по умолчанию 10 Hz):

```text
pose ← local odometry (map frame)
r ← 0
for zone in zones:
  if point_in_polygon(pose.xy, zone.polygon):
    r ← r + zone.reward          # обычно -1 для A, +1 для B
r ← clip(r, r_min, r_max)

# опциональные локальные aversive от собственного тела
if local_collision_event: r ← r + r_collision   # e.g. -0.5
if local_fall_event:      r ← r + r_fall

dan_channel ← 
  aversive  if r < 0
  appetitive if r > 0
  none       otherwise
```

Пластичность:

```text
update W_kc_mbon ONLY where M_dan_mask[dan_channel] == 1
η from connectome runtime config (локально)
```

Соседи **не** добавляют `r` через сеть. Вид соседа — это cue в `u_pn`, не чужой reward.

### 4.3 Карта зон

- Frame: `map` / `arena`.
- Полигоны: список вершин `[[x,y], ...]` в метрах.
- При конфликте зон (перекрытие) — суммирование `reward` с clip (явное в конфиге).

---

## 5. Запрещённые серверные сигналы

Любая реализация, шлющая нижеперечисленное **в контур обучения особи**, нарушает спецификацию.

### 5.1 Жёсткий запрет (MUST NOT)

| Сигнал | Почему нельзя |
|---|---|
| Чужой `r` / `punished` / `rewarded` | Внешний учитель социальности |
| Чужой `action` / `mbon_scores` / `W_out` | Телепатия вместо сенсоров |
| `boost_avoid(cue)` / `social_eta` с PC | Сервер формирует поведение |
| `shared_weights` / federated average online | Социальность вне индивида |
| Peer-to-peer learning messages | То же, минуя PC |
| Live переразметка ценности зон от поведения соседей | Скрытый social reward shaping |

### 5.2 Разрешено снаружи

| Сигнал | Зачем |
|---|---|
| `set_config` + `config_hash` | Условия эпизода |
| `episode_start` / `episode_stop` | Границы опыта |
| `e_stop` / safety limit updates | Безопасность |
| One-way telemetry ingest | Анализ |
| Offline batch jobs на логах | Метрики, графики, ablation |

### 5.3 Серая зона (нужно явно документировать)

- Синхронизация часов (NTP) — ок.
- Раздача одного и того же стартового `connectome_mb_v1.npz` до эпизода — ок (общая *донорская* проводка ≠ внешнее соцповедение).
- Смена зон **между** эпизодами — ок; смена зон mid-episode только как объявленный stimulus protocol, одинаковый для всех, без зависимости от особей.

### 5.4 Тест соответствия

Автоматический или ручной audit:

1. Отключить Wi‑Fi у робота после `episode_start` (кроме локального safety radio, если нужен e-stop).
2. Особь должна продолжать считать `r`, детектить видимых соседей и обновлять веса.
3. Если обучение без линка деградирует до нуля — архитектура не compliant.

E-stop может оставаться на отдельном канале; это не канал обучения.

---

## 6. Локальное восприятие сородичей

Социальные входы — только из onboard perception:

| Cue | Сенсор | PN-маппинг |
|---|---|---|
| `conspecific_visible` | камера + AprilTag/детектор | фиксированный sparse PN-паттерн |
| `conspecific_near` | лидар distance < d_near | threat/spacing PN-пул |
| `conspecific_at_zone_B_seen` | камера: tag в ROI зоны B | отдельный PN-паттерн |
| `looming_conspecific` | быстрый рост bbox / closing lidar | aversive-biased cue |

Флаг `agents[].sense_conspecifics: false` (режим `multi_blind_peers`) **локально** обнуляет эти каналы — контроль для анализа на PC.

---

## 7. Анализ результатов на PC

PC подписан на телеметрию и пишет `episode_id/` лог. Обучение не корректирует.

### 7.1 Обязательные метрики (per agent)

| Метрика | Определение |
|---|---|
| `PI` | `(t_B - t_A) / (t_A + t_B + eps)` за эпизод или окна |
| `time_in_A`, `time_in_B` | секунды в зонах |
| `n_zone_entries` | число входов в A/B |
| `mean_|r|`, `n_plastic_updates` | активность учителя |
| `action_hist` | доля MBON-действий |
| `weight_drift` | `||W_t - W_0||` по masked синапсам |
| `peer_exposure_s` | время с `conspecific_visible` |
| `distance_to_nearest_peer` | распределение |

### 7.2 Популяционные / социальные метрики (post-hoc)

Считаются **только из логов**, не вмешиваются в рантайм:

| Метрика | Смысл |
|---|---|
| `aggregation_index` | скученность относительно random null |
| `avoidance_contagion` | ускорение Avoid_A у особи после *видимого* контакта с особью, часто бывавшей в A (по локальным флагам `*_seen`, не по Wi‑Fi) |
| `blind_vs_sighted_delta` | разница PI / spacing между `multi_independent` и `multi_blind_peers` |
| `flywire_vs_random` | то же для connectome vs `ablation_random_wiring` |
| `individuality` | дисперсия `W` и политик между агентами с одной стартовой проводкой |

Социальный эффект засчитывается **только если** `blind_vs_sighted_delta` значим в пользу sighted при прочих равных условиях арены.

### 7.3 Артефакты анализа

```text
logs/episode_XXX/
  config.yaml
  config.sha256
  agents/go2_i/telemetry.csv
  agents/go2_i/W_kc_mbon_init.npy
  agents/go2_i/W_kc_mbon_final.npy
  agents/go2_i/fly_activations.npz   # optional subsample
  metrics/summary.json
  metrics/figures/...
```

`summary.json` обязан содержать `connectome.manifest_id`, `config_hash`, `mode`, список agent ids.

---

## 8. Минимальные интерфейсы сообщений

### 8.1 PC → Robot (conditions only)

```text
EpisodeConfig      # полный конфиг + hash
EpisodeStart       # episode_id, t0
EpisodeStop        # episode_id, reason
SafetyOverride     # e_stop bool, max_speed optional
```

### 8.2 Robot → PC (observe only)

```text
AgentTelemetry
  stamp, agent_id, pose
  r_self, dan_channel
  action, mbon_ids_active
  cues: zone_flags, conspecific_* (локально вычисленные)
  graph_version
  n_plastic_updates_cum
```

### 8.3 Robot → Robot

Обучающих сообщений нет.
Визуальный AprilTag на корпусе — физический стимул, не сетевой RPC.

---

## 9. Compliance checklist перед экспериментом

- [ ] Стартовый граф = FlyWire artifact с `manifest.json` (snapshot, порог синапсов, типы клеток)
- [ ] `r` считается на роботе из локальной копии зон
- [ ] Нет подписок особи на peer reward/action/weights
- [ ] Social cues только из onboard perception
- [ ] Есть план контролей: `multi_blind_peers`, `ablation_random_wiring`
- [ ] PC пишет логи и считает метрики §7; не шлёт learning hints
- [ ] Wi‑Fi dropout test (§5.4) пройден для одной особи

---

## 10. Краткая формула

> Внешнее задаёт мир и пишет логи.  
> Коннектом + пластичность + восприятие сородичей — только в индивиде.  
> Социальное поведение = выученная особью реакция на сенсорно доступных агентов.
