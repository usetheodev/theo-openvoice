# Theo OpenVoice STT Runtime

## Product Requirements Document (PRD) — v2.1

**Revisão**: v2.1 — clarifica posicionamento: construção from-scratch com bibliotecas maduras. Speaches e Ollama são inspirações, não dependências.
**Changelog v2.1**: Seção "Filosofia de Construção" adicionada. ADR-002 reescrito. Referências a Speaches e Ollama reposicionadas como inspirações arquiteturais. Análise competitiva expandida com Ollama como inspiração de UX/CLI.
**Changelog v2.0**: Audio Preprocessing Pipeline, Inverse Text Normalization, LocalAgreement para partials, Echo Cancellation strategy, Hot Words, Cross-segment context, métricas de qualidade, backpressure WebSocket, corner cases de telefonia.

---

## 1. Visão Geral

Theo OpenVoice STT é o módulo de Speech-to-Text do runtime Theo OpenVoice — a mesma infraestrutura que serve TTS. O objetivo é expor uma **API compatível com OpenAI** para transcrição em tempo real, construída **do zero** como parte do runtime unificado Theo.

O projeto nasce da observação de lacunas recorrentes em projetos open-source existentes no espaço de STT:

- Forte acoplamento a uma única engine (ex: Whisper) sem abstração real.
- Ausência de scheduler temporal consciente de sessões.
- Falta de suporte nativo a cenários de telefonia (ingestão RTP).
- Abstração incompleta para múltiplas arquiteturas de STT.
- **[v2]** Ausência de audio preprocessing pipeline.
- **[v2]** Falta de Inverse Text Normalization (ITN) para output usável em domínios especializados.
- **[v2]** Partial transcripts inviáveis em escala para encoder-decoder sem estratégia otimizada.

Essas lacunas foram observadas em projetos como Speaches, Whisper.cpp, Vosk e LocalAI — que servem como **referência competitiva**, não como base de código. O Theo OpenVoice é construído do zero, usando bibliotecas maduras de inferência (Faster-Whisper, Silero VAD, NeMo) como componentes, não como frameworks.

O STT compartilha com o TTS o mesmo API Server (FastAPI), Model Registry, CLI, observabilidade e infraestrutura de deploy. A diferença está nos workers (engines STT em vez de TTS) e no Session Manager, que é específico de STT.

---

## 2. Filosofia de Construção

### From-Scratch com Bibliotecas Maduras

Theo OpenVoice é um projeto **construído do zero**. Não é fork, wrapper ou extensão de nenhum projeto existente. Todo o runtime — API server, scheduler, session manager, model registry, CLI, pipelines de preprocessing e post-processing — é código original do Theo.

O que **não** reinventamos são as **bibliotecas de inferência e processamento** que já são maduras e testadas:

| Componente | Biblioteca | Papel no Theo |
|---|---|---|
| Inferência STT | Faster-Whisper, WeNet | Engine de inferência, empacotada como worker |
| Voice Activity Detection | Silero VAD | Componente do runtime, orquestrado pelo Session Manager |
| Inverse Text Normalization | nemo_text_processing | Stage do Post-Processing Pipeline |
| Noise Reduction | RNNoise / NSNet2 | Stage do Audio Preprocessing Pipeline |
| Resampling | soxr / scipy | Stage do Audio Preprocessing Pipeline |

Essas bibliotecas são **dependências**, não fundações. O Theo pode trocar qualquer uma delas sem reestruturar o runtime.

### Inspirações Arquiteturais

Dois projetos influenciaram decisões de design do Theo, mas nenhum código deles é utilizado:

**Ollama** — inspiração para o modelo de UX/CLI. A experiência de `ollama pull`, `ollama serve`, `ollama list` para modelos LLM é o padrão que o Theo adapta para modelos de voz: `theo pull`, `theo serve`, `theo list`. A ideia de um registry local com download sob demanda, manifesto declarativo e um único binário que "just works" vem diretamente da observação do Ollama.

**Speaches** — inspiração para o contrato de API. Speaches demonstrou que uma API compatível com OpenAI para STT é viável e desejável. O Theo adota a compatibilidade com a OpenAI Audio API como princípio, mas resolve as limitações arquiteturais observadas: acoplamento ao Whisper, ausência de session manager, falta de scheduler multi-engine, e nenhum suporte a telefonia.

**whisper-streaming** — inspiração para o algoritmo LocalAgreement. O conceito de confirmar tokens por concordância entre passes consecutivas é adaptado do projeto [whisper-streaming](https://github.com/ufal/whisper_streaming) da UFAL. A implementação no Theo é integrada ao Session Manager e ao pipeline de streaming do runtime, não uma cópia do código.

### Princípio Fundamental

> **Construir a camada de runtime que falta no ecossistema.** As engines de inferência existem e são boas. O que não existe é um runtime unificado que orquestre essas engines com session management, preprocessing, post-processing, scheduling, observabilidade e CLI consistente — tudo num único binário que serve STT e TTS.

---

## 3. Relação com o Runtime TTS

### Arquitetura Unificada

```
┌──────────────────────────────────────────────────────────────┐
│                     API Server (FastAPI)                       │
│         /v1/audio/speech (TTS)    /v1/audio/* (STT)           │
├──────────────────────────────────────────────────────────────┤
│                        Scheduler                              │
│              (routing, queue, cancellation)                    │
├──────────────────────────────────────────────────────────────┤
│                     Model Registry                            │
│          (manifesto, lifecycle, eviction)                      │
├──────────────┬───────────────────────────────────────────────┤
│  TTS Workers │            STT Workers                         │
│  (Kokoro,    │  (Faster-Whisper, WeNet,                       │
│   Piper)     │   Paraformer)                                  │
│  subprocess  │  subprocess                                    │
├──────────────┴───────────────────────────────────────────────┤
│              Audio Preprocessing Pipeline [v2]                │
│     (resample, DC remove, normalize, denoise)                 │
├──────────────────────────────────────────────────────────────┤
│              Post-Processing Pipeline [v2]                    │
│     (ITN, entity formatting, hot word correction)             │
├──────────────────────────────────────────────────────────────┤
│                  Session Manager (STT)                         │
│         (estado, ring buffer, VAD, timeout)                    │
└──────────────────────────────────────────────────────────────┘
```

### O que é compartilhado

| Componente | Compartilhado | Notas |
|---|---|---|
| API Server (FastAPI) | Sim | Endpoints TTS e STT no mesmo processo |
| Model Registry | Sim | Mesmo manifesto `theo.yaml`, mesmo lifecycle |
| Scheduler | Sim | Mesmo scheduler, prioridade por tipo (realtime > batch) |
| CLI | Sim | `theo pull`, `theo list`, `theo serve` servem ambos |
| Observabilidade | Sim | Mesmas métricas Prometheus, mesmo `/health` |
| Docker image | Sim | Uma imagem, engines habilitadas por config |
| Audio Preprocessing | Sim | Pipeline compartilhado, stages configuráveis |
| Session Manager | Não | Específico de STT (TTS é stateless por request) |
| Post-Processing | Não | Específico de STT (ITN, entity formatting) |
| Workers | Não | Engines diferentes, protobuf diferente |

### Decisão arquitetural

**Um binário, um processo, dois tipos de worker.** Não existe "Theo STT" e "Theo TTS" como produtos separados. Existe **Theo OpenVoice** com capacidades de TTS e STT, habilitadas por quais modelos estão instalados. Este modelo é inspirado no padrão do Ollama: um único processo que gerencia múltiplos modelos de diferentes tipos.

---

## 4. Análise Competitiva

### Projetos de Referência (Concorrentes)

| Projeto | O que faz | Limitação principal |
|---|---|---|
| **Speaches** | Serve Whisper com API OpenAI-compatible | Acoplado ao Whisper, sem scheduler, sem telefonia |
| **Whisper.cpp server** | Serve Whisper via HTTP | Apenas Whisper, sem streaming real, sem session |
| **whisper-streaming** | LocalAgreement para Whisper online | Sem runtime, sem registry, sem multi-engine |
| **Vosk Server** | Serve Vosk/Kaldi via WebSocket | API proprietária, sem compatibilidade OpenAI |
| **LocalAI** | Runtime genérico (LLM, TTS, STT) | STT é feature secundária, sem session manager |
| **NVIDIA Riva** | STT/TTS enterprise | Proprietário, vendor lock-in NVIDIA |

### Projetos de Inspiração (Não concorrentes)

| Projeto | Inspiração para o Theo | O que NÃO aproveitamos |
|---|---|---|
| **Ollama** | UX/CLI (`pull`, `serve`, `list`), modelo de registry local, single-binary UX | Código, runtime LLM, formato de modelo |
| **Speaches** | Contrato de API compatível com OpenAI, validação de que o approach funciona | Código, arquitetura interna, acoplamento ao Whisper |
| **whisper-streaming** | Algoritmo LocalAgreement para partial transcripts | Código direto, é adaptado e integrado ao Session Manager |

### Diferencial do Theo OpenVoice STT

- **Construído do zero**: runtime original, não fork nem wrapper de projetos existentes.
- **Model-agnostic real**: interface que abstrai encoder-decoder (Whisper), CTC (WeNet), e streaming-native (Paraformer) sem o core assumir tokenizer ou decoder.
- **Session Manager**: estado explícito por sessão com VAD, timeout e recovery — componente que não existe em nenhum projeto open-source de STT.
- **Runtime unificado**: mesmo produto que serve TTS, compartilhando registry, scheduler e CLI.
- **CLI inspirado no Ollama**: `theo pull`, `theo serve`, `theo list` — UX familiar para quem já usa Ollama, adaptado para modelos de voz.
- **Streaming-first com contrato claro**: formato de eventos definido, não apenas "partial transcripts".
- **Preparado para telefonia**: ingestão RTP como módulo, não como hack.
- **[v2] Audio Preprocessing Pipeline**: resample, normalize, denoise — comportamento consistente independente da fonte de áudio.
- **[v2] Post-Processing Pipeline**: ITN e entity formatting plugáveis por idioma/domínio.
- **[v2] Partial transcripts inteligentes**: LocalAgreement (inspirado no whisper-streaming) para encoder-decoder, nativos para CTC/streaming.

---

## 5. Objetivos

**Objetivos principais:**

- Expor API compatível com OpenAI para transcrição (file e streaming).
- Ser model-agnostic: suportar Whisper, CTC e streaming-native sem mudar o core.
- Operar com latência previsível em tempo real (TTFB ≤300ms por segmento).
- Gerenciar sessões de streaming com estado explícito e recovery.
- Compartilhar infraestrutura com o runtime TTS (mesmo binário, registry, scheduler).
- **[v2]** Normalizar áudio de qualquer fonte antes da inferência (preprocessing pipeline).
- **[v2]** Produzir output formatado e usável via post-processing (ITN, entities).
- **[v2]** Suportar hot words / keyword boosting para domínios especializados.

**Não objetivos (v1):**

- Treinamento de modelos.
- UI gráfica.
- SIP signaling (apenas ingestão RTP raw na Fase 3).
- Speaker diarization (escopo futuro).
- Billing / autenticação comercial.
- **[v2]** Acoustic Echo Cancellation (AEC) — responsabilidade do PBX. Documentado como requirement de integração.

---

## 6. Casos de Uso

- **UC-01**: Transcrever arquivo de áudio via REST (`POST /v1/audio/transcriptions`), recebendo texto completo.
- **UC-02**: Transcrever áudio em tempo real via WebSocket, recebendo partial e final transcripts como eventos JSON.
- **UC-03**: Trocar engine STT (ex: Faster-Whisper → WeNet) sem alterar código do cliente — apenas mudar o campo `model`.
- **UC-04**: Manter sessão de streaming por 30+ minutos (call center) com estado gerenciado e recovery de falhas.
- **UC-05**: Receber áudio de um Asterisk via RTP e transcrever em tempo real.
- **UC-06**: Executar múltiplas sessões simultâneas com priorização (telefonia > batch).
- **UC-07**: Traduzir áudio para inglês via endpoint de translation.
- **UC-08 [v2]**: Receber áudio em qualquer sample rate (8kHz, 44.1kHz, 48kHz) e transcrever corretamente via preprocessing automático.
- **UC-09 [v2]**: Transcrever áudio de banking com números formatados corretamente (ex: "dois mil e vinte e cinco" → "2025") via ITN.
- **UC-10 [v2]**: Configurar hot words específicas de domínio (PIX, TED, Selic) para melhorar WER em termos técnicos.
- **UC-11 [v2]**: Transcrever áudio com code-switching (português/inglês) sem forçar single-language decoding.

---

## 7. Requisitos Funcionais

### RF-01: API OpenAI-Compatible para STT

Endpoints compatíveis com OpenAI Audio API (contrato inspirado na validação do Speaches de que essa API funciona para STT):

- `POST /v1/audio/transcriptions` — transcrição de arquivo.
- `POST /v1/audio/translations` — tradução para inglês.
- `WS /v1/realtime` — streaming bidirecional em tempo real (Fase 2).

### RF-02: Transcrição de Arquivo (Batch)

Upload de arquivo de áudio, processamento completo, retorno de texto. Suporte aos formatos: WAV, MP3, FLAC, OGG, WebM.

**[v2] Batched inference**: Para engines que suportam (Faster-Whisper `BatchedInferencePipeline`), batch processing de segments dentro de um arquivo para aceleração de 2-3x em GPU.

### RF-03: Streaming STT

Transcrição em tempo real via WebSocket com:

- Partial transcripts (hipóteses intermediárias).
- Final transcripts (segmentos confirmados).
- Eventos de VAD (voice activity start/end).
- Detecção de silêncio com commit automático de segmento.

**Especificações de streaming de entrada:**

- Formato de áudio aceito na engine: PCM 16-bit, 16kHz, mono (raw).
- **[v2] Formato de áudio aceito do cliente**: qualquer sample rate. O Audio Preprocessing Pipeline faz resample para 16kHz automaticamente.
- Tamanho de window de processamento: 30ms por frame de entrada.
- **[v2] Window de VAD**: 64ms (1024 samples a 16kHz) — acumular 2 frames de 30ms + padding de 4ms para melhor acurácia do Silero VAD.
- Envio: cliente envia frames de áudio como mensagens binárias WebSocket.
- **[v2] Tamanho máximo de mensagem WebSocket**: 64KB (equivalente a ~2s de PCM 16kHz mono).
- **[v2] Tamanho recomendado de frame**: 20ms ou 40ms (padrão de telefonia).
- Amostra mínima para processamento: 500ms de áudio acumulado.

### RF-04: Voice Activity Detection (VAD)

VAD opera no nível do runtime (não da engine), usando Silero VAD como biblioteca:

- Threshold configurável (default: 0.5).
- Min speech duration: 250ms.
- Min silence duration: 300ms (para commit de segmento).
- Max speech duration: 30s (force commit).
- **[v2] VAD sensitivity levels**: `high | normal | low` — ajusta threshold e pre-filter de energia conjuntamente.

| Sensitivity | Threshold | Energy Pre-filter | Caso de uso |
|---|---|---|---|
| `high` | 0.3 | -50dBFS | Banking confidencial, fala sussurrada |
| `normal` | 0.5 | -40dBFS | Default, conversação normal |
| `low` | 0.7 | -30dBFS | Ambientes muito ruidosos, call center |

**[v2] Energy-based Pre-filter (antes do Silero VAD):**

Pre-filtro baseado em energia e spectral flatness para reduzir falsos positivos do Silero VAD em ambientes ruidosos:

1. Calcular RMS do frame.
2. Se RMS < threshold dBFS configurado E spectral flatness > 0.8 (ruído branco), classificar como silêncio sem invocar Silero.
3. Caso contrário, passar para Silero VAD.

Custo: ~0.1ms por frame. Redução estimada de falsos positivos: 60-70% em ambientes ruidosos.

**[v2] Limitação documentada**: Silero VAD não detecta whispered speech de forma confiável. Para cenários bancários/confidenciais, recomendar `vad_sensitivity: high`.

VAD no runtime garante comportamento consistente entre engines.

### RF-05: Seleção Dinâmica de Modelo

Cada request especifica o modelo via campo `model`. O runtime resolve para o backend STT correto e carrega se necessário. Mesmo mecanismo do TTS — inspirado no modelo de registry do Ollama onde `ollama run <model>` resolve, baixa e carrega o modelo sob demanda.

### RF-06: Session Manager

Gerenciamento explícito de sessões de streaming com estados definidos:

```
INIT → ACTIVE → SILENCE → HOLD → CLOSING → CLOSED
                  ↑          ↑        |
                  └──────────┘────────┘  (nova fala detectada)
```

**Estados:**

| Estado | Descrição | Timeout |
|---|---|---|
| `INIT` | Sessão criada, aguardando primeiro áudio | **[v2] 30s** (sem áudio → CLOSED). Configurável via `session.configure`. |
| `ACTIVE` | Recebendo áudio com fala detectada | — |
| `SILENCE` | VAD detectou silêncio, aguardando retomada | 30s (configurável, → HOLD se > 30s sem fala) |
| `HOLD` **[v2]** | Silêncio prolongado (ex: chamada em hold) | 5min (configurável, → CLOSING) |
| `CLOSING` | Flush de partial transcripts pendentes | 2s (→ CLOSED) |
| `CLOSED` | Sessão encerrada, recursos liberados | — |

**[v2] Justificativa do estado HOLD**: Em call centers, chamadas em hold (música de espera, transferência) podem durar 1-10 minutos. O estado SILENCE com timeout de 30s fecharia a sessão prematuramente. HOLD mantém a sessão aberta com recursos mínimos (ring buffer ativo, worker em idle).

**Recovery:**

Se worker crashar durante sessão ACTIVE, o Session Manager:

1. Detecta falha via **[v2] gRPC stream break** (detecção imediata, não polling).
2. Emite evento `error` ao cliente com `recoverable: true`.
3. Reinicia worker.
4. Retoma sessão do último segmento confirmado (final transcript).
5. Áudio no ring buffer entre o último commit e o crash é reprocessado.

**[v2] Write-Ahead Log (WAL) para recovery:**

Antes de enviar `transcript.final` ao cliente, o runtime registra em memória:

- `last_committed_segment_id`
- `last_committed_buffer_offset`
- `last_committed_timestamp_ms`

No recovery, o Session Manager usa esses ponteiros para retomar sem duplicação. O WAL é in-memory (não disco) para não impactar latência.

**[v2] Ring Buffer Read Fence:**

O ring buffer mantém um `read_fence` no `last_committed_offset`. Dados antes do fence podem ser sobrescritos; dados depois, não. Se o buffer atingir 90% de capacidade sem commit, o runtime força commit do segmento atual (mesmo sem VAD silence) para liberar espaço.

### RF-07: Cancelamento

Cliente pode cancelar sessão via WebSocket `close` ou mensagem `cancel`. Runtime propaga cancelamento ao worker em ≤50ms. Partial transcripts pendentes são descartados.

### RF-08: Multi-idioma

Detecção automática de idioma (se engine suportar) ou seleção explícita via campo `language` na request.

**[v2] Code-switching**: Suporte a `language: "mixed"` ou `language: ["pt", "en"]` que indica ao runtime para não forçar single-language decoding. Para Whisper, `language: "mixed"` traduz para `language=None` (auto-detect per segment).

### RF-09 [v2]: Audio Preprocessing Pipeline

Pipeline de pré-processamento de áudio entre ingestão e VAD/engine. Cada stage é toggleável via config. Construído como componente original do runtime, usando bibliotecas de DSP (soxr, scipy, RNNoise) como dependências.

```
Ingestão (WebSocket/RTP) → Resample → DC Remove → Gain Normalize → [Denoise] → VAD → Engine
```

**Stages:**

| Stage | Descrição | Default | Custo |
|---|---|---|---|
| **Resample** | Converte qualquer sample rate para 16kHz mono. Usa `soxr` (alta qualidade) ou `scipy.signal.resample_poly` | Ativo | <1ms/frame |
| **DC Remove** | High-pass filter a 20Hz para remover DC offset de hardware (comum em telefonia) | Ativo | <0.1ms/frame |
| **Gain Normalize** | Normaliza amplitude para range consistente (-3dBFS peak). Essencial para CTC models sensíveis a amplitude | Ativo | <0.1ms/frame |
| **Denoise** | Noise reduction via RNNoise ou NSNet2. Reduz ruído de fundo antes do VAD e engine | **Desativo** (habilitar para telefonia) | ~1ms/frame CPU |

**Total do pipeline**: <5ms por frame em CPU (com denoise habilitado).

**Configuração:**

```yaml
preprocessing:
  resample: true
  dc_remove: true
  gain_normalize: true
  target_dbfs: -3.0
  denoise: false           # habilitar para telefonia
  denoise_engine: rnnoise  # rnnoise | nsnet2
```

### RF-10 [v2]: Post-Processing Pipeline

Pipeline de pós-processamento entre output da engine e resposta ao cliente. Plugável por idioma e domínio. Componente original do runtime que orquestra bibliotecas especializadas.

```
Engine Output → ITN → Entity Formatting → Hot Word Correction → Response
```

**Stages:**

| Stage | Descrição | Biblioteca |
|---|---|---|
| **Inverse Text Normalization (ITN)** | "dois mil e vinte e cinco" → "2025", "dez por cento" → "10%" | `nemo_text_processing` (NeMo, suporta pt-BR) |
| **Entity Formatting** | CPF: "um dois três ponto..." → "123.456.789-00", valores monetários | Custom rules por domínio (código original Theo) |
| **Hot Word Correction** | Corrige transcrições próximas de hot words configurados. Ex: "pics" → "PIX" | Levenshtein distance + boost (código original Theo) |

**Configuração:**

```yaml
post_processing:
  itn:
    enabled: true
    language: pt
  entity_formatting:
    enabled: false
    domain: banking    # banking | medical | legal | generic
  hot_word_correction:
    enabled: false
```

### RF-11 [v2]: Hot Words / Keyword Boosting

Suporte a hot words configuráveis por sessão para melhorar WER em termos de domínio específico.

**Comportamento por arquitetura:**

| Architecture | Mecanismo de Hot Words |
|---|---|
| `encoder-decoder` (Whisper) | Injetados via `initial_prompt`. Ex: `"Termos: PIX, TED, Selic, CDI."` |
| `ctc` (WeNet) | Keyword boosting nativo via decoding parameters |
| `streaming-native` (Paraformer) | Hot word list nativa (se suportado) |

**Configuração via API:**

```json
{
  "type": "session.configure",
  "hot_words": ["PIX", "TED", "Selic", "CDI", "IPCA"],
  "hot_word_boost": 2.0
}
```

Para batch:

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

file=@audio.wav
model=faster-whisper-large-v3
hot_words=PIX,TED,Selic
```

### RF-12 [v2]: Cross-Segment Context

Para sessões longas, o runtime envia contexto do segmento anterior como `initial_prompt` para o próximo segmento. Isso melhora continuidade em palavras cortadas no limite de segmentos.

**Implementação:**

- Armazenar últimos N tokens (default: 224 tokens, metade do context window do Whisper) do `transcript.final` mais recente.
- Enviar como `initial_prompt` na próxima inferência.
- Aplicável apenas a engines que suportam conditioning (Whisper). Para CTC/streaming-native, ignorar.

---

## 8. Requisitos Não Funcionais

### RNF-01: Latência

- **Batch (arquivo)**: tempo total proporcional à duração do áudio. Fator alvo: ≤0.5x (30s de áudio processado em ≤15s).
- **[v2] Batch com BatchedInferencePipeline**: fator alvo ≤0.2x em GPU (30s processado em ≤6s).
- **Streaming TTFB**: ≤300ms do recebimento de segmento com fala até primeiro partial transcript.
- **Final transcript delay**: ≤500ms após VAD detectar fim de fala.
- **[v2] Preprocessing overhead**: ≤5ms total por frame (incluindo denoise).
- **[v2] Post-processing overhead**: ≤10ms por segmento final.
- **Medição**: do momento em que o runtime acumula áudio suficiente até o primeiro evento de transcript ser escrito no WebSocket.

### RNF-02: Estabilidade de Sessão

- Sessões de streaming devem suportar ≥30 minutos contínuos sem degradação de latência ou memory leak.
- Ring buffer com tamanho fixo (default: 60s de áudio, ~1.9MB em PCM 16kHz mono).
- **[v2] Ring buffer com read fence**: dados antes do last_committed_offset podem ser sobrescritos; dados depois, protegidos.
- **[v2] Force commit**: se ring buffer atingir 90% sem commit, forçar commit do segmento atual.

### RNF-03: Isolamento de Falhas

- Crash de worker STT não afeta o runtime, TTS workers, nem outras sessões STT.
- Restart automático de worker com recovery de sessão (RF-06).
- **[v2] Detecção de falha via gRPC stream break** (imediata) em vez de health check polling.

### RNF-04: Uso de GPU

- Múltiplas sessões compartilham GPU via batching no worker (se engine suportar).
- Sem reallocations de CUDA durante inferência (pre-allocate buffers no load).
- Fallback transparente para CPU.
- **[v2] Concurrency model explícito**:
  - Fase 1: 1 worker = 1 sessão ativa (batch ou streaming). Escala horizontal com mais workers.
  - Fase 2: Dynamic batching no worker (estilo NVIDIA Triton): acumula requests por até 50ms, batch inference, distribui resultados.
- **[v2] Referência de capacidade**: A10G com Whisper large-v3 suporta ~4 sessões batch simultâneas ou ~2 sessões streaming com partials.

### RNF-05: Observabilidade

Métricas Prometheus adicionais (além das compartilhadas com TTS):

**Métricas operacionais:**

- `theo_stt_ttfb_seconds` — tempo até primeiro partial transcript.
- `theo_stt_final_delay_seconds` — delay do final transcript após fim de fala.
- `theo_stt_active_sessions` — sessões ativas.
- `theo_stt_session_duration_seconds` — histograma de duração de sessões.
- `theo_stt_vad_events_total` — eventos VAD (speech_start, speech_end).
- `theo_stt_worker_errors_total` — erros por worker.
- `theo_stt_preprocessing_duration_seconds` — tempo de preprocessing por frame.
- `theo_stt_postprocessing_duration_seconds` — tempo de postprocessing por segmento.

**[v2] Métricas de qualidade:**

- `theo_stt_confidence_avg` — confidence média dos final transcripts (proxy para WER).
- `theo_stt_no_speech_segments_total` — segmentos retornados como empty/no_speech. Valor alto indica VAD misconfiguration.
- `theo_stt_language_detection_mismatches_total` — quando `language` explícito difere do detectado pela engine.
- `theo_stt_hot_word_hits_total` — quantas vezes hot words configurados apareceram no output. Valor zero persistente indica hot words ineficazes.
- `theo_stt_segments_force_committed_total` — segmentos com force commit (ring buffer cheio). Valor alto indica `max_segment_duration` muito alto ou silêncio não detectado.

### RNF-06 [v2]: Backpressure WebSocket

- Se o cliente enviar áudio mais rápido que real-time (ex: arquivo via WebSocket), o runtime aplica throttling.
- Emite evento `session.rate_limit` com `delay_ms` sugerindo ao cliente quanto esperar.
- Buffer máximo de backlog: 10s de áudio. Acima disso, frames são descartados com evento `session.frames_dropped`.

### RNF-07 [v2]: Heartbeat WebSocket

- Server envia ping WebSocket a cada 10s.
- Se pong não recebido em 5s, sessão transita para CLOSING.
- Previne connections zombies em redes instáveis (mobile, WiFi).

---

## 9. Contrato de API

### POST /v1/audio/transcriptions

**Request (multipart/form-data):**

| Campo | Tipo | Obrigatório | Default | Descrição |
|---|---|---|---|---|
| `file` | binary | sim | — | Arquivo de áudio |
| `model` | string | sim | — | Identificador do modelo no registry |
| `language` | string | não | auto-detect | Código ISO 639-1. **[v2]** `"mixed"` para code-switching |
| `prompt` | string | não | — | Contexto para guiar transcrição |
| `response_format` | string | não | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | não | `0.0` | Temperatura de sampling (0.0-1.0) |
| `timestamp_granularities` | array | não | `["segment"]` | `["word"]`, `["segment"]`, ou ambos |
| `hot_words` | string | não | — | **[v2]** Lista separada por vírgula: `"PIX,TED,Selic"` |
| `itn` | bool | não | `true` | **[v2]** Aplicar Inverse Text Normalization ao output |

**Response (`json`):**

```json
{
  "text": "Olá, como posso ajudar?"
}
```

**Response (`verbose_json`):**

```json
{
  "task": "transcribe",
  "language": "pt",
  "duration": 2.5,
  "text": "Olá, como posso ajudar?",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Olá, como posso ajudar?",
      "tokens": [1234, 5678],
      "temperature": 0.0,
      "avg_logprob": -0.25,
      "compression_ratio": 1.1,
      "no_speech_prob": 0.01
    }
  ],
  "words": [
    { "word": "Olá", "start": 0.0, "end": 0.5 },
    { "word": "como", "start": 0.6, "end": 0.9 },
    { "word": "posso", "start": 1.0, "end": 1.3 },
    { "word": "ajudar", "start": 1.4, "end": 2.5 }
  ]
}
```

**Erros:**

| Status | Significado |
|---|---|
| `400` | Formato de áudio não suportado ou input inválido |
| `404` | Modelo não encontrado no registry |
| `413` | Arquivo excede limite (default: 25MB) |
| `503` | Modelo em loading (cold start) |

### POST /v1/audio/translations

Mesmo contrato que `/transcriptions`, mas output sempre em inglês. Campo `language` indica o idioma do áudio de entrada (opcional, auto-detect).

### WS /v1/realtime (Fase 2)

**Handshake:**

```
GET /v1/realtime?model=faster-whisper-large-v3&language=pt
Upgrade: websocket
```

**Mensagens Client → Server:**

```json
// Enviar áudio (mensagem binária)
// Frames PCM 16-bit, qualquer sample rate (runtime faz resample)
// Recomendado: 20ms ou 40ms de frame size
// Max message size: 64KB
// Enviar como binary WebSocket messages

// Configurar sessão
{
  "type": "session.configure",
  "vad_threshold": 0.5,
  "vad_sensitivity": "normal",
  "silence_timeout_ms": 300,
  "hold_timeout_ms": 300000,
  "max_segment_duration_ms": 30000,
  "language": "pt",
  "hot_words": ["PIX", "TED", "Selic", "CDI"],
  "hot_word_boost": 2.0,
  "enable_partial_transcripts": true,
  "enable_itn": true,
  "preprocessing": {
    "denoise": true,
    "denoise_engine": "rnnoise"
  },
  "input_sample_rate": 8000
}

// Cancelar
{
  "type": "session.cancel"
}

// Commit manual de segmento
{
  "type": "input_audio_buffer.commit"
}

// Fechar sessão
{
  "type": "session.close"
}
```

**Mensagens Server → Client:**

```json
// Sessão criada
{
  "type": "session.created",
  "session_id": "sess_abc123",
  "model": "faster-whisper-large-v3",
  "config": {
    "vad_threshold": 0.5,
    "vad_sensitivity": "normal",
    "silence_timeout_ms": 300,
    "hold_timeout_ms": 300000,
    "preprocessing": { "resample": true, "denoise": true }
  }
}

// VAD: fala detectada
{
  "type": "vad.speech_start",
  "timestamp_ms": 1500
}

// Partial transcript
{
  "type": "transcript.partial",
  "text": "Olá como",
  "segment_id": 0,
  "timestamp_ms": 2000
}

// Final transcript
{
  "type": "transcript.final",
  "text": "Olá, como posso ajudar?",
  "segment_id": 0,
  "start_ms": 1500,
  "end_ms": 4000,
  "language": "pt",
  "confidence": 0.95,
  "words": [
    { "word": "Olá", "start": 1.5, "end": 2.0 },
    { "word": "como", "start": 2.1, "end": 2.4 }
  ]
}

// VAD: silêncio detectado
{
  "type": "vad.speech_end",
  "timestamp_ms": 4000
}

// [v2] Sessão em hold
{
  "type": "session.hold",
  "timestamp_ms": 34000,
  "hold_timeout_ms": 300000
}

// [v2] Backpressure
{
  "type": "session.rate_limit",
  "delay_ms": 100,
  "message": "Client sending faster than real-time, please throttle"
}

// [v2] Frames descartados
{
  "type": "session.frames_dropped",
  "dropped_ms": 500,
  "message": "Backlog exceeded 10s, frames dropped"
}

// Erro (recuperável)
{
  "type": "error",
  "code": "worker_crash",
  "message": "Worker restarted, resuming from segment 5",
  "recoverable": true,
  "resume_segment_id": 5
}

// Sessão encerrada
{
  "type": "session.closed",
  "reason": "client_request",
  "total_duration_ms": 45000,
  "segments_transcribed": 12
}
```

---

## 10. CLI (Extensão)

Comandos adicionais ao CLI compartilhado. O modelo de CLI segue o padrão popularizado pelo Ollama (`pull`, `serve`, `list`), adaptado para modelos de voz (STT e TTS):

```bash
# Comandos compartilhados (inspirados no Ollama)
theo pull faster-whisper-large-v3       # Baixa modelo STT
theo pull kokoro-v1                      # Baixa modelo TTS
theo list                                # Lista modelos instalados (STT + TTS)
theo serve                               # Inicia runtime (serve todos os modelos instalados)
theo ps                                  # Lista modelos carregados em memória
theo remove faster-whisper-large-v3      # Remove modelo
theo inspect faster-whisper-large-v3     # Detalhes do modelo

# Comandos STT
theo transcribe <file> --model faster-whisper-large-v3   # Transcreve arquivo
theo transcribe <file> --model faster-whisper-large-v3 --format srt  # Gera legenda
theo transcribe --stream --model faster-whisper-large-v3  # Streaming do microfone
theo translate <file> --model faster-whisper-large-v3     # Traduz para inglês
theo transcribe <file> --hot-words "PIX,TED,Selic"        # [v2] Com hot words
theo transcribe <file> --no-itn                            # [v2] Sem ITN
```

---

## 11. Model Registry — Manifesto STT

Extensão do mesmo `theo.yaml` usado pelo TTS. O registry local segue o conceito do Ollama de manifestos declarativos que descrevem modelos, suas capabilities e requirements:

```yaml
name: faster-whisper-large-v3
version: 3.0.0
engine: faster-whisper
type: stt
description: "Faster Whisper Large V3 - encoder-decoder STT"

capabilities:
  streaming: true
  architecture: encoder-decoder
  languages: ["auto", "en", "pt", "es", "ja", "zh"]
  sample_rate: 16000
  word_timestamps: true
  translation: true
  partial_transcripts: true
  hot_words: false              # [v2] suporta keyword boosting nativo?
  batch_inference: true         # [v2] suporta batch mode?
  language_detection: true      # [v2] detecta idioma automaticamente?
  initial_prompt: true          # [v2] suporta conditioning via prompt?

quality:                         # [v2]
  telephony_optimized: false     # treinado/testado com áudio 8kHz?
  wer_benchmark: 0.045          # WER em LibriSpeech test-clean (referência)

resources:
  memory_mb: 3072
  gpu_required: false
  gpu_recommended: true
  load_time_seconds: 8

engine_config:
  model_size: "large-v3"
  compute_type: "float16"
  device: "auto"
  beam_size: 5
  vad_filter: false  # VAD é feito no runtime, não na engine

preprocessing:                   # [v2]
  target_sample_rate: 16000
  normalize_audio: true
  noise_reduction: false
```

**Campo `type`**: `stt` ou `tts`. O registry usa esse campo para rotear requests ao tipo correto de worker.

**Campo `architecture`**: informa o runtime sobre o modelo de inferência:

| Architecture | Exemplos | Implicações |
|---|---|---|
| `encoder-decoder` | Whisper, Distil-Whisper | Processa chunks acumulados. **[v2] Partials via LocalAgreement.** |
| `ctc` | WeNet CTC, Wav2Vec2 CTC | Character/token-level output, baixa latência |
| `streaming-native` | Paraformer, WeNet streaming | Streaming verdadeiro, partial transcripts nativos |

O runtime adapta o comportamento de windowing e partial transcript com base na architecture declarada.

**[v2] Campo `quality.telephony_optimized`**: O registry usa esse campo para recomendar modelos adequados quando o audio source é telefonia (8kHz). Se o modelo não é telephony_optimized e o áudio é 8kHz, o runtime emite warning no log e sugere modelo alternativo (se disponível).

---

## 12. Interface de Backend STT

Cada backend STT implementa a seguinte interface Python. Esta interface é o contrato original do Theo para integrar engines de inferência — é o que torna o runtime model-agnostic:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum


class STTArchitecture(Enum):
    ENCODER_DECODER = "encoder-decoder"
    CTC = "ctc"
    STREAMING_NATIVE = "streaming-native"


@dataclass
class TranscriptSegment:
    """Segmento de transcrição (partial ou final)."""
    text: str
    is_final: bool
    segment_id: int
    start_ms: int | None = None
    end_ms: int | None = None
    language: str | None = None
    confidence: float | None = None
    words: list[dict] | None = None  # [{"word": "olá", "start": 0.0, "end": 0.5}]


@dataclass
class BatchResult:
    """Resultado de transcrição batch (arquivo completo)."""
    text: str
    language: str
    duration: float
    segments: list[dict]
    words: list[dict] | None = None


@dataclass
class EngineCapabilities:
    """[v2] Capabilities reportadas pela engine em runtime."""
    supports_hot_words: bool = False
    supports_initial_prompt: bool = False
    supports_batch: bool = False
    supports_word_timestamps: bool = False
    max_concurrent_sessions: int = 1


class STTBackend(ABC):
    """Interface que todo backend STT deve implementar."""

    @property
    @abstractmethod
    def architecture(self) -> STTArchitecture:
        """Retorna a arquitetura do modelo."""
        ...

    @abstractmethod
    async def load(self, model_path: str, config: dict) -> None:
        """Carrega o modelo em memória."""
        ...

    @abstractmethod
    async def capabilities(self) -> EngineCapabilities:
        """[v2] Retorna capabilities da engine em runtime (pode diferir do manifesto)."""
        ...

    @abstractmethod
    async def transcribe_file(
        self,
        audio_path: str,
        language: str | None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        **kwargs,
    ) -> BatchResult:
        """Transcreve arquivo de áudio completo (batch)."""
        ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        language: str | None,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[TranscriptSegment]:
        """
        Transcreve áudio em streaming.

        Recebe: AsyncIterator de chunks PCM 16-bit 16kHz mono
                (já preprocessado pelo Audio Preprocessing Pipeline do runtime).
        Produz: TranscriptSegment (partial e final).

        Para encoder-decoder: [v2] runtime usa LocalAgreement para partials.
        Para streaming-native: a engine processa frame a frame.
        Para CTC: a engine produz character-level output progressivo.
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Descarrega o modelo da memória."""
        ...

    @abstractmethod
    async def health(self) -> dict:
        """Retorna status do backend."""
        ...
```

### Adaptação por Arquitetura

O runtime adapta o pipeline de streaming baseado na `architecture` declarada:

**Encoder-decoder (Whisper) — [v2] com LocalAgreement:**

```
Audio frames → Preprocessing → Ring Buffer → LocalAgreement Policy
                                                     ↓
                                        Accumulate window (2-5s)
                                                     ↓
                                        Engine.transcribe_stream()
                                                     ↓
                                        Compare with previous pass
                                                     ↓
                              Tokens concordantes → transcript.partial (confirmed prefix)
                              Tokens novos → aguardar próxima pass
                                                     ↓
                              VAD silence → transcript.final (flush all)
```

**[v2] LocalAgreement (adaptado do conceito do whisper-streaming):**

O algoritmo LocalAgreement resolve o problema de partial transcripts para encoder-decoder sem custo proibitivo. O conceito é inspirado no projeto [whisper-streaming](https://github.com/ufal/whisper_streaming) da UFAL, com implementação própria integrada ao Session Manager e pipeline do Theo:

1. Roda inference em windows incrementais (não re-processa tudo).
2. Compara output da pass atual com a anterior.
3. Tokens que aparecem em ambas as passes no mesmo offset são confirmados como `partial`.
4. Tokens que ainda não convergiram são retidos.
5. Custo: ~1 inferência por window (vs. 2/s no approach naive).

**Configuração:**

```yaml
partial_strategy:
  encoder_decoder: local_agreement  # local_agreement | naive_reprocess | disabled
  local_agreement_window_ms: 3000   # window de cada pass
  local_agreement_min_confirm: 2    # passes mínimas para confirmar token
```

**[v2] Fallback**: Se `partial_strategy: disabled` para encoder-decoder, o runtime emite apenas `transcript.final` após VAD silence. Isso é o comportamento mais eficiente em GPU e recomendado para batch-heavy workloads.

**CTC:**

```
Audio frames → Preprocessing → Ring Buffer → Engine.transcribe_stream() (frame by frame)
                                                        ↓
                                              TranscriptSegment (partial + final)
```

**Streaming-native:**

```
Audio frames → Preprocessing → Engine.transcribe_stream() (frame by frame, engine gerencia estado)
                                              ↓
                                    TranscriptSegment (partial + final, nativos)
```

---

## 13. Comunicação Runtime ↔ Worker (gRPC)

Protocolo gRPC original do Theo para comunicação entre o runtime e os workers de inferência:

```protobuf
service STTWorker {
  // Transcrição batch (arquivo)
  rpc TranscribeFile (TranscribeFileRequest) returns (TranscribeFileResponse);

  // Transcrição streaming (bidirecional)
  rpc TranscribeStream (stream AudioFrame) returns (stream TranscriptEvent);

  // Cancelamento
  rpc Cancel (CancelRequest) returns (CancelResponse);

  // Health (unário — mas detecção de crash usa stream break)
  rpc Health (HealthRequest) returns (HealthResponse);
}

message TranscribeFileRequest {
  string request_id = 1;
  bytes audio_data = 2;
  string language = 3;
  string response_format = 4;
  float temperature = 5;
  repeated string timestamp_granularities = 6;
  string initial_prompt = 7;        // [v2] contexto/conditioning
  repeated string hot_words = 8;    // [v2] keyword boosting
}

message TranscribeFileResponse {
  string text = 1;
  string language = 2;
  float duration = 3;
  repeated Segment segments = 4;
  repeated Word words = 5;
}

message AudioFrame {
  string session_id = 1;
  bytes data = 2;          // PCM 16-bit 16kHz mono (já preprocessado)
  bool is_last = 3;
  string initial_prompt = 4;    // [v2] contexto do segmento anterior
  repeated string hot_words = 5; // [v2]
}

message TranscriptEvent {
  string session_id = 1;
  string type = 2;         // "partial", "final"
  string text = 3;
  int32 segment_id = 4;
  int64 start_ms = 5;
  int64 end_ms = 6;
  string language = 7;
  float confidence = 8;
  repeated Word words = 9;
}

message Segment {
  int32 id = 1;
  float start = 2;
  float end = 3;
  string text = 4;
  float avg_logprob = 5;
  float no_speech_prob = 6;
}

message Word {
  string word = 1;
  float start = 2;
  float end = 3;
}
```

**[v2] Detecção de crash via stream break**: O gRPC bidirecional stream (`TranscribeStream`) funciona como heartbeat implícito. Se o worker crashar, o stream termina com erro. O runtime detecta imediatamente (vs. polling de health a cada 1s). O endpoint `Health` unário é mantido para status checks de tooling (CLI, monitoring).

---

## 14. Ring Buffer e Windowing

### Ring Buffer

Buffer circular de tamanho fixo que armazena áudio recente da sessão. Componente original do runtime:

- **Tamanho default**: 60s de áudio (1,920,000 bytes em PCM 16-bit 16kHz mono).
- **Propósito**: permite reprocessamento após recovery de falha, e acumulação de windows para encoder-decoder.
- **Implementação**: array pré-alocado com ponteiros de read/write. Zero-copy quando possível.
- **[v2] Read fence**: ponteiro `last_committed_offset` que marca o ponto até onde o áudio já foi processado e confirmado (final transcript enviado). Áudio antes do fence pode ser sobrescrito; áudio depois do fence é protegido.
- **[v2] Force commit trigger**: se o buffer atingir 90% de capacidade com dados não commitados, o runtime emite `force_commit` ao Session Manager, que finaliza o segmento atual independente do VAD.

### Windowing por Arquitetura

| Arquitetura | Window size | Overlap | Reprocessa window? | Partial strategy [v2] |
|---|---|---|---|---|
| encoder-decoder | 3-5s (acumulado até VAD silence) | 0 | Não (com LocalAgreement) | LocalAgreement |
| ctc | 30ms (frame a frame) | 0 | Não | Nativo |
| streaming-native | 30ms (frame a frame) | Engine-defined | Não | Nativo |

---

## 15. Backends Iniciais

### Faster-Whisper

- Engine: CTranslate2 (C++ com bindings Python). **Usado como biblioteca de inferência**, não como runtime.
- Modelos: Whisper tiny → large-v3, Distil-Whisper.
- Arquitetura: encoder-decoder.
- Streaming: **[v2] via LocalAgreement no runtime** (implementação original do Theo, inspirada no conceito do whisper-streaming).
- **[v2] Batch**: suporta `BatchedInferencePipeline` para 2-3x speedup em arquivos.
- Ideal para: qualidade máxima de transcrição, batch, GPU.
- **[v2] Nota sobre Distil-Whisper**: Distil-large-v3 é ~6x mais rápido que large-v3 com ~1% de degradação em WER. Recomendado para cenários de streaming onde latência é prioridade sobre accuracy máxima.
- **[v2] Nota sobre telephony**: Whisper perde ~5-15% de accuracy com áudio 8kHz upsampled. Para telefonia pura, considerar modelos fine-tuned ou backends otimizados para narrowband.

### WeNet (Fase 2)

- Engine: Python/C++ com LibTorch. **Usado como biblioteca de inferência**.
- Modelos: CTC e attention-based.
- Arquitetura: ctc / streaming-native.
- Streaming: nativo.
- **[v2] Hot words**: suporte nativo a keyword boosting via decoding parameters.
- Ideal para: latência ultra-baixa, streaming real.

### [v2] Modelos Recomendados por Cenário

| Cenário | Modelo recomendado | Justificativa |
|---|---|---|
| Batch, qualidade máxima | faster-whisper-large-v3 | Melhor WER geral |
| Streaming, baixa latência | distil-whisper-large-v3 | 6x mais rápido, ~1% WER gap |
| Streaming, ultra-baixa latência | WeNet CTC / Paraformer | Partials nativos, <100ms TTFB |
| Telefonia 8kHz | WeNet telephony / Whisper fine-tuned | Otimizados para narrowband |
| CPU-only | faster-whisper-tiny / WeNet CTC | Modelos leves |

---

## 16. Audio Preprocessing Pipeline — Detalhes [v2]

### Arquitetura

Componente original do runtime Theo. Usa bibliotecas de DSP como dependências, mas a orquestração, configuração e integração com o pipeline de streaming é código próprio.

```
┌─────────────────────────────────────────────────────┐
│              Audio Preprocessing Pipeline             │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ Resample │→ │ DC Remove│→ │ Gain Norm│→ │Denoise│ │
│  │ (soxr)   │  │ (HPF 20Hz)│  │(-3dBFS)  │  │(RNN) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                                       │
│  Input: any sample rate, any format                   │
│  Output: PCM 16-bit, 16kHz, mono, normalized          │
└─────────────────────────────────────────────────────┘
```

### Implementação por stage

**Resample**: Usa `soxr` (via `soxr-python`) para resampling de alta qualidade. Alternativa: `scipy.signal.resample_poly` (menor dependência). Detecta sample rate do input automaticamente (via header WAV ou campo `input_sample_rate` no WebSocket config).

**DC Remove**: Butterworth high-pass filter de 2ª ordem, fc=20Hz. Remove DC offset sem afetar fala (banda útil: 80Hz-8kHz). Implementação: `scipy.signal.sosfilt` com coeficientes pré-calculados no init da sessão.

**Gain Normalize**: Peak normalization para -3dBFS. Calcula fator de ganho por window de 500ms (não por frame, para evitar pumping). Clipping protection: se qualquer sample exceder 0dBFS após ganho, limitar.

**Denoise**: RNNoise (C com bindings Python via `rnnoise-python`) ou NSNet2 (ONNX). Habilitado apenas quando `preprocessing.denoise: true`. Latência: ~1ms/frame em CPU. Melhora WER significativamente em áudio ruidoso (telefonia, viva-voz).

### Configuração global (theo.yaml)

```yaml
server:
  preprocessing:
    resample: true
    target_sample_rate: 16000
    dc_remove: true
    dc_remove_cutoff_hz: 20
    gain_normalize: true
    target_dbfs: -3.0
    normalize_window_ms: 500
    denoise: false
    denoise_engine: rnnoise   # rnnoise | nsnet2
```

### Configuração por sessão (WebSocket)

```json
{
  "type": "session.configure",
  "preprocessing": {
    "denoise": true,
    "denoise_engine": "rnnoise"
  },
  "input_sample_rate": 8000
}
```

---

## 17. Post-Processing Pipeline — Detalhes [v2]

### Arquitetura

Componente original do runtime Theo. ITN usa `nemo_text_processing` como biblioteca; Entity Formatting e Hot Word Correction são implementações próprias.

```
┌─────────────────────────────────────────────────────┐
│              Post-Processing Pipeline                 │
│                                                       │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   ITN    │→ │Entity Format │→ │Hot Word Correct  │ │
│  │ (NeMo)  │  │  (Theo)      │  │  (Theo)          │ │
│  └──────────┘  └──────────────┘  └─────────────────┘ │
│                                                       │
│  Input: raw transcript text                           │
│  Output: formatted text                               │
└─────────────────────────────────────────────────────┘
```

### Inverse Text Normalization (ITN)

Converte palavras numéricas para dígitos e formata entidades:

| Input | Output |
|---|---|
| "dois mil e vinte e cinco" | "2025" |
| "dez por cento" | "10%" |
| "quinze reais e cinquenta centavos" | "R$15,50" |
| "zero vinte e um nove oito meia" | "021 9 8 6" |

**Biblioteca**: `nemo_text_processing` (NVIDIA NeMo). Suporta pt-BR nativamente. Alternativa: `itn` library ou custom regex rules.

**Nota**: ITN é aplicado apenas em `transcript.final`, nunca em `transcript.partial` (partials são instáveis e ITN poderia gerar output confuso).

### Entity Formatting (por domínio)

Regras adicionais de formatação específicas do domínio. **Implementação própria do Theo** — regras configuráveis via JSON:

**Banking:**
- CPF: "um dois três quatro cinco seis sete oito nove zero zero" → "123.456.789-00"
- CNPJ: similar
- Agência/conta: "agência zero um dois três" → "agência 0123"
- Valores: "cento e cinquenta mil" → "R$150.000,00"

**Medical:**
- CID: "cid dez jota zero seis" → "CID-10 J06"
- Dosagens: "quinhentos miligramas" → "500mg"

**Configuração**: domínio selecionado via config. Rules são arquivos JSON carregáveis em runtime.

### Hot Word Correction

Pós-correção baseada em distância de edição. **Implementação própria do Theo:**

1. Para cada word no output, calcular Levenshtein distance contra hot words configurados.
2. Se distance ≤ threshold (default: 2) E phonetic similarity alta, substituir.
3. Ex: "pics" → "PIX" (distance=2), "selique" → "Selic" (distance=2).

**Nota**: Isso é um fallback. O mecanismo primário de hot words opera no nível da engine (initial_prompt para Whisper, keyword boosting para CTC). A correção pós-engine pega o que a engine errou.

---

## 18. Ingestão RTP (Fase 3)

### Escopo

**Apenas ingestão de RTP raw.** O SIP signaling é responsabilidade do PBX (Asterisk/FreeSWITCH). Theo recebe o stream de áudio já decodificado.

### Fluxo

```
Asterisk ──RTP (G.711/PCM)──→ Theo RTP Listener ──→ Codec Decode ──→ Preprocessing ──→ Session Manager
```

### Componente: RTP Listener

Componente original do runtime Theo:

- Recebe pacotes UDP com payload RTP.
- Extrai áudio, aplica jitter buffer (20ms, configurável).
- Decodifica G.711 μ-law/A-law para PCM 16-bit.
- **[v2] Nota de qualidade**: G.711 é 8kHz, 8-bit. O Audio Preprocessing Pipeline faz upsample para 16kHz automaticamente, mas a qualidade é limitada a 4kHz de bandwidth. O runtime deve sinalizar `audio_quality: telephony` ao registry para selecionar modelos otimizados.
- Alimenta Audio Preprocessing Pipeline (que faz resample, normalize, etc.).

### O que NÃO faz

- SIP INVITE/BYE (Asterisk gerencia).
- Echo cancellation (Asterisk gerencia).
- DTMF detection (Asterisk gerencia).
- Media negotiation (Asterisk gerencia).

### [v2] Echo Cancellation — Requirement de Integração

**Problema**: Em cenários full-duplex (STT + TTS simultâneos), o microfone captura o áudio do TTS. Sem echo cancellation, o STT transcreve o que o bot disse.

**Decisão**: AEC é responsabilidade do PBX (Asterisk/FreeSWITCH), NÃO do Theo. Porém, isso deve ser explicitamente documentado como **requirement de integração**.

**Requirements para Asterisk:**

1. `TALK_DETECT` habilitado no dialplan para detectar barge-in.
2. Echo cancellation habilitado no channel driver (PJSIP: `echo_cancel=yes`, `echo_cancel_tail_length=200`).
3. Se usando confbridge/MixMonitor, garantir que o áudio enviado ao Theo via RTP é o áudio do caller **apenas** (sem mix com TTS output).
4. Alternativa: enviar reference signal (TTS output) como segundo stream RTP para que o Theo possa fazer reference subtraction. Isso é um **extensão opcional da Fase 3+**.

**Fallback sem AEC**: Se AEC não está disponível, o runtime pode usar **mute-on-speak**: pausar ingestão STT enquanto TTS está ativo na mesma sessão. Simples mas elimina barge-in.

---

## 19. Roadmap

### Fase 1 — STT Batch + Preprocessing (6 semanas)

**Objetivo**: API funcional para transcrição de arquivo, construída do zero com Faster-Whisper como biblioteca de inferência.

**Entregáveis:**

- `POST /v1/audio/transcriptions` — endpoint construído no API Server FastAPI do Theo.
- `POST /v1/audio/translations` para tradução.
- Worker Faster-Whisper: subprocess com gRPC, usando Faster-Whisper como **biblioteca de inferência** (não fork/wrapper de Speaches).
- Formato de resposta: `json`, `verbose_json`, `text`, `srt`, `vtt`.
- **[v2] Audio Preprocessing Pipeline**: resample, DC remove, gain normalize (denoise desabilitado por default).
- **[v2] Post-Processing Pipeline**: ITN básico via nemo_text_processing.
- **[v2] BatchedInferencePipeline**: batch processing para speedup em GPU.
- Model registry compartilhado com TTS (campo `type: stt`).
- CLI: `theo transcribe <file>`, `theo translate <file>` (seguindo padrão Ollama do CLI existente).
- Testes de latência batch.
- **[v2] Métricas de qualidade**: confidence_avg, no_speech_segments.

**Critério de sucesso**: `curl -F file=@audio.wav -F model=faster-whisper-large-v3 http://localhost:8000/v1/audio/transcriptions` retorna transcrição correta com números formatados (ITN).

**Estratégia**: Construir worker STT do zero, usando Faster-Whisper como biblioteca de inferência da mesma forma que se usa qualquer dependência Python. O valor está no runtime original (registry, scheduler, preprocessing, post-processing, CLI unificado), não em reescrever inference. As engines de inferência são excelentes — o que falta é o runtime que as orquestra.

### Fase 2 — Streaming Real + Session Manager (8 semanas)

**Entregáveis:**

- `WS /v1/realtime` com protocolo de eventos definido (original do Theo, inspirado na estrutura da OpenAI Realtime API).
- Session Manager com estados (INIT → ACTIVE → SILENCE → **HOLD** → CLOSING → CLOSED) — componente original sem equivalente em projetos open-source de STT.
- VAD via Silero VAD (como biblioteca) no runtime **[v2] com energy-based pre-filter** (implementação própria).
- **[v2] VAD sensitivity levels** (high/normal/low).
- Ring buffer com **[v2] read fence e force commit** — implementação original.
- **[v2] LocalAgreement** para partial transcripts de encoder-decoder — implementação própria, inspirada no conceito do whisper-streaming.
- **[v2] Cross-segment context** via initial_prompt.
- Partial e final transcripts via WebSocket.
- **[v2] Backpressure e heartbeat WebSocket.**
- **[v2] Hot words** via session.configure.
- **[v2] WAL in-memory** para session recovery sem duplicação.
- CLI: `theo transcribe --stream`.
- Segundo backend STT (WeNet) demonstrando model-agnostic — validando que a interface `STTBackend` suporta arquiteturas fundamentalmente diferentes.

**Critério de sucesso**: sessão WebSocket de 30 minutos sem degradação de latência, com recovery de falha de worker sem duplicação de segmentos.

### Fase 3 — Telefonia + Scheduler Avançado (8 semanas)

**Entregáveis:**

- RTP Listener (componente original) com jitter buffer e decode G.711.
- **[v2] Preprocessing automático**: detect 8kHz, upsample, denoise habilitado por default para RTP.
- **[v2] Audio quality tagging**: sinalizar `audio_quality: telephony` ao registry.
- Integração testada com Asterisk (receber áudio de chamada, transcrever em tempo real).
- **[v2] Documentação de integration requirements**: AEC, TALK_DETECT, channel isolation.
- **[v2] Mute-on-speak fallback** para cenários sem AEC.
- Scheduler com priorização: realtime (WebSocket/RTP) > batch (file upload).
- Orçamento de latência por sessão no scheduler.
- Co-scheduling STT + TTS (para agentes full-duplex).
- **[v2] Dynamic batching** no worker (estilo Triton): acumula requests, batch inference, distribui.

---

## Architecture Decision Records (ADRs)

---

### ADR-001 — Runtime Unificado STT + TTS

**Status:** Aceito

**Decisão:** STT e TTS são módulos do mesmo runtime (Theo OpenVoice), não produtos separados.

**Justificativa:**

- Compartilham 70%+ da infraestrutura: API server, registry, scheduler, CLI, observabilidade, Docker.
- Um binário simplifica deploy e operação.
- Agentes de voz precisam de ambos — oferecer num único `theo serve` é o UX correto.
- Model registry com campo `type` diferencia naturalmente.
- Modelo inspirado no Ollama: um único processo que gerencia múltiplos tipos de modelo.

**Trade-off:** complexidade do binário único aumenta, mas é compensada pela eliminação de duplicação.

---

### ADR-002 — Construção From-Scratch com Bibliotecas Maduras

**Status:** Aceito

**Decisão:** O Theo OpenVoice é construído do zero. Projetos como Speaches e Ollama são inspirações arquiteturais, não dependências ou base de código. Bibliotecas maduras de inferência (Faster-Whisper, Silero VAD, NeMo) são usadas como componentes substituíveis.

**Justificativa:**

- O valor do Theo está no **runtime** — a camada de orquestração que não existe no ecossistema: session manager, model registry, scheduler multi-engine, preprocessing/post-processing pipelines, CLI unificado.
- As engines de inferência já existem e são excelentes. Reimplementar inference seria desperdício.
- Usar bibliotecas maduras como dependências (não como frameworks) mantém a arquitetura desacoplada — qualquer engine pode ser substituída sem reestruturar o runtime.
- Speaches validou que a API compatível com OpenAI funciona para STT — o Theo adota o mesmo contrato de API, mas com runtime completamente diferente.
- Ollama validou que o modelo `pull/serve/list` com registry local é a UX correta para modelos de AI — o Theo adapta esse padrão para modelos de voz.

**O que é construído do zero:**

- API Server (FastAPI) com endpoints STT e TTS
- Scheduler com priorização e cancelamento
- Session Manager com estados, recovery e WAL
- Model Registry com manifesto declarativo e lifecycle
- Audio Preprocessing Pipeline (orquestração dos stages)
- Post-Processing Pipeline (ITN orchestration, Entity Formatting, Hot Word Correction)
- Ring Buffer com read fence e force commit
- LocalAgreement para partial transcripts (inspirado no conceito, implementação própria)
- RTP Listener com jitter buffer
- CLI (`theo pull`, `theo serve`, `theo transcribe`, etc.)
- Protocolo gRPC de comunicação runtime ↔ worker
- Protocolo de eventos WebSocket para streaming

**O que é usado como biblioteca (dependência substituível):**

| Biblioteca | Papel | Substituível por |
|---|---|---|
| Faster-Whisper | Engine de inferência STT | WeNet, Paraformer, qualquer `STTBackend` |
| Silero VAD | Voice Activity Detection | Qualquer VAD que aceite PCM 16kHz |
| nemo_text_processing | Inverse Text Normalization | itn library, custom regex |
| RNNoise / NSNet2 | Noise reduction | Qualquer denoiser que processe PCM frames |
| soxr / scipy | Resampling | Qualquer resampler com qualidade equivalente |
| Kokoro / Piper | Engine de inferência TTS | Qualquer `TTSBackend` |

**Alternativas consideradas:**

- Fork do Speaches: descartado. Speaches é acoplado ao Whisper, sem session manager, sem scheduler, sem multi-engine. Forkar adicionaria dívida técnica significativa para depois remover o acoplamento.
- Extensão do LocalAI: descartado. STT é feature secundária no LocalAI, arquitetura não suporta session management.
- Wrapper sobre Ollama: descartado. Ollama é focado em LLMs, modelos de voz têm requirements diferentes (streaming, VAD, preprocessing).

---

### ADR-003 — VAD no Runtime, Não na Engine

**Status:** Aceito (com extensão v2)

**Decisão:** Voice Activity Detection roda no runtime (usando Silero VAD como biblioteca), não dentro da engine STT.

**Justificativa:**

- Comportamento consistente entre engines diferentes.
- Engines como Whisper têm VAD próprio mas com comportamento inconsistente.
- Runtime controla o pipeline de segmentação independente do modelo.
- Silero VAD é leve (~2ms por frame) e roda em CPU sem impacto.

**[v2] Extensão:**

- **Energy-based pre-filter** (implementação própria) antes do Silero para reduzir falsos positivos em ambientes ruidosos.
- **Sensitivity levels** (high/normal/low) para adaptar a diferentes cenários.
- **Limitação documentada**: Silero VAD é fraco com whispered speech. Mitigação: sensitivity `high`.
- **Window size**: VAD roda em windows de 64ms (não 30ms) para melhor acurácia.

**Trade-off:** engines com VAD nativo superior (se houver) não aproveitam sua vantagem. Configurável via manifesto (`vad_filter: true` na engine_config permite VAD dual).

---

### ADR-004 — Windowing Adaptativo por Arquitetura (com LocalAgreement)

**Status:** Aceito (revisado v2)

**Decisão:** O runtime adapta o pipeline de streaming baseado na `architecture` declarada no manifesto. **[v2] Para encoder-decoder, usa LocalAgreement (implementação própria inspirada no conceito do whisper-streaming) em vez de re-processamento naive.**

**Justificativa original:**

- Encoder-decoder (Whisper) precisa acumular áudio antes de processar. Streaming é simulado.
- CTC processa frame a frame com output incremental.
- Streaming-native (Paraformer) tem pipeline próprio.
- Uma interface única (`transcribe_stream`) com comportamento adaptado é melhor que forçar todas as engines no mesmo modelo.

**[v2] Revisão: LocalAgreement substitui re-processamento naive:**

O approach original (re-processar window a cada 500ms) é inviável em escala:
- Whisper Large-v3 em GPU (A10G): ~200ms para processar 5s de áudio.
- Re-processar a cada 500ms = 2 inferências/s por sessão.
- 10 sessões = 20 inferências/s → satura A10G.
- Em CPU: impossível.

LocalAgreement resolve:
- 1 inferência por window (vs 2/s).
- Confirma tokens que concordam entre passes consecutivas.
- Qualidade de partials comparável, custo ~50% menor.
- Conceito validado pelo projeto [whisper-streaming](https://github.com/ufal/whisper_streaming). Implementação no Theo é própria, integrada ao Session Manager.

**Fallback**: partial desabilitado para encoder-decoder é opção válida e mais eficiente. Documentado como configuração.

---

### ADR-005 — Ring Buffer Pre-alocado (com Read Fence)

**Status:** Aceito (com extensão v2)

**Decisão:** Cada sessão tem um ring buffer de tamanho fixo (default 60s) pré-alocado. **[v2] Com read fence para proteção de dados não commitados.**

**Justificativa original:**

- Elimina allocations durante streaming (requisito para latência previsível).
- Permite reprocessamento após recovery de falha.
- 60s × 16kHz × 2 bytes = 1.9MB por sessão — custo aceitável.
- Tamanho fixo previne memory leak em sessões longas.

**[v2] Extensão — Read Fence:**

Problema identificado: se o ring buffer sobrescreve dados não commitados (monólogo de 55s com ring buffer de 60s), dados se perdem na recovery.

Solução: `read_fence` no `last_committed_offset`. Dados antes do fence podem ser sobrescritos. Dados depois, protegidos. Se buffer atingir 90% sem commit, force commit do segmento atual.

**[v2] Extensão — WAL in-memory:**

Problema identificado: race condition se worker crashar enquanto emite final transcript. Possível duplicação de segmento.

Solução: WAL in-memory com `last_committed_segment_id`, `last_committed_buffer_offset`, `last_committed_timestamp_ms`. Runtime verifica no recovery para retomar sem duplicação.

---

### ADR-006 — Protocolo de Eventos WebSocket (com extensões v2)

**Status:** Aceito (estendido v2)

**Decisão:** Protocolo de eventos JSON original do Theo, inspirado na estrutura da API Realtime da OpenAI mas simplificado para STT-only. **[v2] Com eventos adicionais para backpressure, hold, e health.**

**Justificativa original:**

- Compatibilidade conceitual com OpenAI facilita adoção (desenvolvedores já conhecem o padrão).
- JSON legível facilita debugging.
- Tipos de evento explícitos são mais claros que SSE genérico.
- Overhead de JSON parse é negligível comparado à inferência.
- Protocolo é original do Theo — inspirado na estrutura da OpenAI, mas com eventos e semântica próprios.

**[v2] Eventos adicionais:**

| Evento | Direção | Descrição |
|---|---|---|
| `session.hold` | Server→Client | Sessão transitou para estado HOLD |
| `session.rate_limit` | Server→Client | Backpressure — cliente enviando mais rápido que real-time |
| `session.frames_dropped` | Server→Client | Frames descartados por excesso de backlog (>10s) |
| Ping/Pong WebSocket | Bidirecional | Heartbeat a cada 10s, timeout de pong em 5s |

**[v2] Framing protocol:**

- Frame size recomendado: 20ms ou 40ms.
- Max message size: 64KB.
- Heartbeat: server ping a cada 10s.

**Subset implementado (vs OpenAI Realtime):**

| OpenAI Realtime | Theo | Status |
|---|---|---|
| `session.create` | `session.configure` | Simplificado |
| `input_audio_buffer.append` | Binary WebSocket message | Simplificado |
| `input_audio_buffer.commit` | `input_audio_buffer.commit` | Igual |
| `response.audio_transcript.delta` | `transcript.partial` | Renomeado |
| `conversation.item.created` | Não implementado | Fora de escopo |
| Response/turn management | Não implementado | Fora de escopo (é LLM) |

---

### ADR-007 — Escopo de Telefonia: Apenas RTP Raw (com AEC requirements)

**Status:** Aceito (estendido v2)

**Decisão:** Na Fase 3, Theo implementa apenas ingestão RTP raw. SIP signaling fica fora. **[v2] AEC é responsabilidade do PBX, documentado como integration requirement.**

**Justificativa original:**

- SIP é um protocolo complexo. Implementar é um projeto em si.
- Asterisk/FreeSWITCH já resolvem SIP de forma madura.
- Focar em RTP raw mantém escopo controlável e testável.

**[v2] AEC decision:**

- AEC é essencial para full-duplex (STT + TTS simultâneos).
- Implementar AEC no Theo adicionaria complexidade significativa (speexdsp, referência de TTS, latência).
- Asterisk já tem AEC no channel driver (PJSIP).
- **Decisão: documentar como requirement**, não implementar.
- **Fallback: mute-on-speak** para cenários sem AEC (sacrifica barge-in).

---

### ADR-008 [v2] — Audio Preprocessing no Runtime

**Status:** Aceito

**Decisão:** Audio preprocessing (resample, DC remove, normalize, denoise) é pipeline do runtime (implementação original), usando bibliotecas de DSP como dependências. Não é responsabilidade da engine.

**Justificativa:**

- Engines esperam input normalizado (PCM 16-bit, 16kHz, mono).
- Clientes enviam áudio em qualquer formato (8kHz telefonia, 44.1kHz desktop, 48kHz WebRTC).
- Sem preprocessing, cada engine implementaria (ou não) sua própria normalização, resultando em comportamento inconsistente.
- CTC models são especialmente sensíveis a amplitude — gain normalize é essencial.
- Denoise antes do VAD reduz falsos positivos significativamente.
- Pipeline configurável permite trade-off latência vs qualidade por cenário.

**Trade-off:** Adiciona ~5ms de latência por frame. Aceitável dado o benefício em robustez e consistência.

---

### ADR-009 [v2] — LocalAgreement para Partial Transcripts

**Status:** Aceito

**Decisão:** Para engines encoder-decoder (Whisper), usar algoritmo LocalAgreement (implementação própria, inspirada no conceito do whisper-streaming) para partial transcripts em vez de re-processamento naive de window.

**Justificativa:**

- Re-processamento naive consome 2+ inferências/s por sessão — inviável em escala.
- LocalAgreement usa 1 inferência por window com comparação entre passes.
- Tokens confirmados (concordância entre passes) são emitidos como partial.
- Custo ~50% menor que naive approach.
- Qualidade de partials comparável.
- Conceito validado pelo projeto whisper-streaming (UFAL). Implementação no Theo é original, integrada ao Session Manager e ring buffer.

**Trade-off:** Adiciona complexidade ao runtime (manter estado entre passes). Justificado pelo ganho de eficiência.

**Alternativas consideradas:**

- `partial_strategy: disabled` — mais eficiente, mas sem partials. Válido como configuração.
- `partial_strategy: naive_reprocess` — mantido como opção para debugging/teste, mas não recomendado em produção.
- Usar whisper-streaming diretamente: descartado. Não se integra ao Session Manager, ring buffer, e pipeline de recovery do Theo. O conceito é aproveitado, a implementação é própria.

---

### ADR-010 [v2] — Post-Processing Pipeline Plugável

**Status:** Aceito

**Decisão:** Post-processing (ITN, entity formatting, hot word correction) é pipeline do runtime, plugável por idioma e domínio. ITN usa nemo_text_processing como biblioteca; Entity Formatting e Hot Word Correction são implementações originais do Theo.

**Justificativa:**

- STT genérico produz output não usável para domínios especializados (banking, medical).
- "dois mil e vinte e cinco" precisa virar "2025" para ser útil.
- Formatação de entidades (CPF, CNPJ, valores) é domain-specific — não pertence à engine.
- Pipeline plugável permite adicionar domínios sem modificar engine ou runtime core.
- `nemo_text_processing` oferece ITN para pt-BR de qualidade — não faz sentido reimplementar.
- Entity Formatting e Hot Word Correction são domain-specific do Theo — implementação própria.

**Trade-off:** Adiciona ~10ms por segmento final. Aceitável. ITN errors são possíveis — monitorar via métricas de qualidade.

---

## Encerramento

Theo OpenVoice STT v2.1 é um runtime de voz **construído do zero**, com a visão de preencher a lacuna que existe no ecossistema open-source entre engines de inferência excelentes e a camada de runtime que as orquestra em produção.

**O que construímos (código original):**

- Runtime unificado STT + TTS com API Server, Scheduler, Session Manager
- Model Registry com manifesto declarativo e lifecycle (padrão inspirado no Ollama)
- Audio Preprocessing Pipeline e Post-Processing Pipeline
- Ring Buffer com read fence, WAL in-memory, force commit
- LocalAgreement para partial transcripts (conceito inspirado no whisper-streaming)
- RTP Listener para ingestão de telefonia
- CLI unificado (padrão inspirado no Ollama)
- Protocolo gRPC runtime ↔ worker e protocolo WebSocket de streaming
- Métricas de qualidade e observabilidade

**O que usamos como bibliotecas (dependências substituíveis):**

- Faster-Whisper, WeNet, Paraformer (engines de inferência)
- Silero VAD (voice activity detection)
- nemo_text_processing (inverse text normalization)
- RNNoise / NSNet2 (noise reduction)
- soxr / scipy (audio DSP)

**O que nos inspira (sem código compartilhado):**

- Ollama — UX de CLI e modelo de registry local
- Speaches — validação de API compatível com OpenAI para STT
- whisper-streaming — conceito de LocalAgreement para partial transcripts

A estratégia é clara: **construir a camada de runtime que falta**, usando as melhores bibliotecas disponíveis como componentes, e os melhores projetos open-source como inspiração — sem forkar, sem wrappear, sem atalhos.