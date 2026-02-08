# Theo OpenVoice -- Plano Estrategico do Milestone M4

**Versao**: 1.0
**Base**: ROADMAP.md v1.0, PRD v2.1, ARCHITECTURE.md v1.0
**Status**: **Concluido** (2026-02-08)
**Data**: 2026-02-08

**Autora**: Sofia Castellani (Principal Solution Architect)

---

## 1. Contexto e Objetivo

M4 e o milestone que completa a Fase 1 do PRD. Ate agora, o Theo transcreve arquivos de audio via REST API e CLI -- mas o audio vai cru para o worker (sem normalizacao) e o texto volta cru da engine (sem formatacao). Um audio em 44.1kHz e enviado diretamente ao Faster-Whisper que espera 16kHz. Um usuario que diz "dois mil e quinhentos reais" recebe `"dois mil e quinhentos reais"` em vez de `"R$2.500,00"`.

M4 fecha essas duas lacunas com dois pipelines:

1. **Audio Preprocessing Pipeline**: normaliza audio de qualquer fonte (resample, DC remove, gain normalize) antes de enviar ao worker.
2. **Post-Processing Pipeline**: formata texto da engine (ITN via nemo_text_processing) antes de retornar ao cliente.

### O que ja existe (M1 + M2 + M3)

```
src/theo/
  _types.py              -> BatchResult, ResponseFormat, etc.
  exceptions.py          -> AudioFormatError, InvalidRequestError, etc.
  logging.py             -> structlog (JSON + console)
  config/
    manifest.py          -> ModelManifest (parsing de theo.yaml)
    preprocessing.py     -> PreprocessingConfig (ja definido, campos prontos)
    postprocessing.py    -> PostProcessingConfig, ITNConfig (ja definido, campos prontos)
  server/
    app.py               -> create_app() factory
    routes/_common.py    -> handle_audio_request() (ponto de integracao)
    formatters.py        -> format_response() (BatchResult -> json/srt/vtt)
  scheduler/
    scheduler.py         -> Scheduler.transcribe() (recebe audio_data, envia ao worker)
  registry/
    registry.py          -> ModelRegistry (scan, get_manifest, list_models)
  workers/
    manager.py           -> WorkerManager (spawn, health, restart)
    stt/
      interface.py       -> STTBackend ABC
      faster_whisper.py  -> FasterWhisperBackend (transcribe_file)
      servicer.py        -> STTWorkerServicer (gRPC)
  cli/
    transcribe.py        -> theo transcribe (thin HTTP client)
  preprocessing/
    __init__.py          -> (vazio -- pacote criado, sem implementacao)
  postprocessing/
    __init__.py          -> (vazio -- pacote criado, sem implementacao)
```

### O que M4 entrega

```
ANTES (M3):  Audio vai raw ao worker. Texto volta raw ao cliente.
             curl -F file=@audio_44khz.wav -> {"text": "dois mil e quinhentos reais"}

DEPOIS (M4): Audio e normalizado (16kHz, mono, -3dBFS). Texto e formatado (ITN).
             curl -F file=@audio_44khz.wav -> {"text": "2.500 reais"}
             theo transcribe audio.wav --no-itn -> {"text": "dois mil e quinhentos reais"}
```

### Principio do M4

> **Dois pipelines compostos por stages independentes.** Cada stage segue uma interface minima (`process(data) -> data`), e toggleavel via config, e testavel isoladamente. Os pipelines se encaixam no fluxo existente sem mudar o contrato da API nem a interface do Scheduler.

### Conexao com o Roadmap

M4 e o ultimo milestone do Tema T2 (Transcricao Batch). Apos M4, a Fase 1 do PRD esta 100% entregue:

```
PRD Fase 1 (STT Batch) ✅
  M1: Fundacao          [x] Completo
  M2: Worker gRPC       [x] Completo
  M3: API Batch + CLI   [x] Completo
  M4: Pipelines         [x] Completo
```

### O que muda para o usuario apos M4

1. **Audio de qualquer sample rate funciona corretamente.** Um arquivo em 44.1kHz (gravacao desktop), 8kHz (telefonia), ou 48kHz (WebRTC) e normalizado automaticamente para 16kHz antes da inferencia.
2. **Numeros e entidades sao formatados.** "dois mil e vinte e cinco" vira "2025". "dez por cento" vira "10%".
3. **Controle do usuario.** Flag `--no-itn` no CLI e campo `itn=false` na API permitem desabilitar formatacao quando o texto cru e preferido.

---

## 2. Decisoes de Design

### 2.1 Pontos de Integracao no Fluxo Existente

Os pipelines se inserem no fluxo batch existente sem alterar interfaces:

```
FLUXO M3 (atual):
  Upload -> handle_audio_request() -> scheduler.transcribe(audio_data) -> Worker -> BatchResult -> format_response()

FLUXO M4 (novo):
  Upload -> handle_audio_request() -> [PREPROCESSING] -> scheduler.transcribe(audio_processado) -> Worker -> BatchResult -> [POSTPROCESSING] -> format_response()
```

**Ponto de integracao do preprocessing**: dentro de `handle_audio_request()` em `server/routes/_common.py`, entre a leitura do arquivo (`audio_data = await file.read()`) e a criacao do `TranscribeRequest`. O audio bruto passa pelo pipeline de preprocessing e o resultado (PCM 16kHz mono normalizado) e o que vai no `TranscribeRequest.audio_data`.

**Ponto de integracao do post-processing**: dentro de `handle_audio_request()`, entre `result = await scheduler.transcribe(request)` e `return format_response(result, fmt)`. O `BatchResult.text` (e os `segments[].text`) passam pelo pipeline de post-processing (ITN) antes de serem formatados.

### 2.2 Interface dos Stages

**Preprocessing (audio -> audio):**

```python
class AudioStage(ABC):
    """Stage do pipeline de preprocessamento de audio."""

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Processa frame de audio.

        Args:
            audio: Array numpy com samples (mono, float32 ou int16).
            sample_rate: Sample rate atual do audio.

        Returns:
            Tupla (audio processado, novo sample rate).
        """
        ...
```

**Post-processing (texto -> texto):**

```python
class TextStage(ABC):
    """Stage do pipeline de pos-processamento de texto."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Processa texto.

        Args:
            text: Texto cru da engine.

        Returns:
            Texto processado.
        """
        ...
```

Interfaces minimas. KISS. Sem genericos, sem metadados, sem side-effects. Cada stage recebe dados, retorna dados transformados.

### 2.3 Pipeline como Orquestrador

O pipeline e uma lista ordenada de stages, cada um habilitado/desabilitado via config:

```python
class AudioPreprocessingPipeline:
    """Orquestra stages de preprocessamento de audio."""

    def __init__(self, config: PreprocessingConfig) -> None: ...
    def process(self, audio_bytes: bytes) -> bytes: ...
    #            ^ raw do upload         ^ PCM 16kHz mono normalizado

class PostProcessingPipeline:
    """Orquestra stages de pos-processamento de texto."""

    def __init__(self, config: PostProcessingConfig) -> None: ...
    def process(self, text: str) -> str: ...
```

### 2.4 O que NAO fazer (fronteiras YAGNI)

| Item | Por que nao | Quando |
|------|-------------|--------|
| Stage Denoise (RNNoise) | Desativado por default no PRD. Binding Python instavel. Dependencia pesada. | M8 (telefonia, onde e necessario) |
| Entity Formatting (CPF, CNPJ, valores) | Domain-specific, requer regras configuráveis por dominio. Sem caso de uso concreto agora. | Futuro (quando houver cliente banking) |
| Hot Word Correction (Levenshtein) | Depende de hot words configurados por sessao. Batch nao tem sessao. | M6 (Session Manager + streaming) |
| Preprocessing em modo streaming (frame-by-frame) | M4 e batch-only. Projetar interface para suportar streaming futuro, mas implementar apenas batch. | M5 (WebSocket + VAD) |
| Metricas Prometheus (`preprocessing_duration_seconds`, `postprocessing_duration_seconds`) | Metricas sao valiosas, mas nao bloqueiam entrega funcional. | Task separada no final de M4 |

**Nota sobre streaming-readiness**: Conforme perspectiva do Viktor no ROADMAP.md, a interface dos stages de preprocessing DEVE ser projetada para operar frame-by-frame desde o inicio. O `AudioStage.process()` recebe um frame (numpy array), nao um arquivo completo. Em M4, o pipeline lida com arquivo completo decompondo em frames internamente. Em M5, o mesmo pipeline recebe frames do WebSocket.

### 2.5 Dependencia `nemo_text_processing`

**Risco identificado**: `nemo_text_processing` e uma dependencia pesada que puxa PyTorch como dependencia transitiva.

**Mitigacao**:
- `nemo_text_processing` ja esta declarado como extra opcional no `pyproject.toml`: `itn = ["nemo_text_processing>=1.1,<2.0"]`
- O stage ITN verifica se o pacote esta disponivel no import. Se nao, fallback graceful: retorna texto sem transformacao e emite warning no log.
- Testes unitarios do stage ITN usam mock do `nemo_text_processing`. Testes de integracao requerem o pacote instalado.
- Flag `itn=false` na API e `--no-itn` no CLI permitem desabilitar completamente.

---

## 3. Epics e Tasks

### Grafo de Dependencias

```
E1-T1 (AudioStage interface + AudioPreprocessingPipeline)
  +---> E1-T2 (Stage Resample)
  |       +---> E1-T3 (Stage DC Remove)
  |               +---> E1-T4 (Stage Gain Normalize)
  |                       +---> E1-T5 (Integracao preprocessing no fluxo batch)
  |
E2-T1 (TextStage interface + PostProcessingPipeline)
  +---> E2-T2 (Stage ITN)
          +---> E2-T3 (Integracao post-processing no fluxo batch)
                  +---> E3-T1 (CLI: --no-itn, API: itn param)
                          +---> E3-T2 (Testes e2e pipeline completo)
```

### Sequencia de implementacao recomendada

A ordem maximiza valor incremental. Apos cada task, algo novo funciona:

```
Fase A -- Preprocessing Pipeline
  1. E1-T1  Interface AudioStage + pipeline orquestrador    [algo instanciavel]
  2. E1-T2  Stage Resample (soxr/scipy)                     [44kHz -> 16kHz funciona]
  3. E1-T3  Stage DC Remove (HPF 20Hz)                      [DC offset removido]
  4. E1-T4  Stage Gain Normalize (-3dBFS)                    [amplitude normalizada]
  5. E1-T5  Integracao no fluxo batch                        [curl com audio 44kHz funciona!]

Fase B -- Post-Processing Pipeline
  6. E2-T1  Interface TextStage + pipeline orquestrador      [algo instanciavel]
  7. E2-T2  Stage ITN (nemo_text_processing)                 ["dois mil" -> "2000"]
  8. E2-T3  Integracao no fluxo batch                        [curl retorna texto formatado!]

Fase C -- Controle e Validacao
  9.  E3-T1  CLI --no-itn + API itn param                   [usuario controla ITN]
  10. E3-T2  Testes e2e pipeline completo                    [validacao end-to-end]
```

---

### Epic 1 -- Audio Preprocessing Pipeline

#### E1-T1: Interface AudioStage + AudioPreprocessingPipeline

**Contexto/Motivacao**: Antes de implementar qualquer stage individual, precisamos do esqueleto: a interface que todos os stages seguem e o orquestrador que os executa em sequencia. Sem isso, cada stage seria implementado de forma ad-hoc sem composabilidade.

**Escopo**:

| In | Out |
|----|-----|
| ABC `AudioStage` com metodo `process(audio, sample_rate) -> (audio, sample_rate)` | Stages concretos (T2, T3, T4) |
| `AudioPreprocessingPipeline` com `process(audio_bytes) -> bytes` | Modo streaming (frame-by-frame via WebSocket) |
| Pipeline aceita lista de stages habilitados via `PreprocessingConfig` | Denoise stage |
| Conversao `bytes` (raw upload) -> `numpy.ndarray` no ponto de entrada | |
| Conversao `numpy.ndarray` -> `bytes` (PCM 16-bit) no ponto de saida | |
| Deteccao de sample rate do audio de entrada (via header WAV ou fallback) | |
| Teste unitario com stage identity (passthrough) | |

**Entregaveis**:
- `src/theo/preprocessing/stages.py` -- ABC `AudioStage`
- `src/theo/preprocessing/pipeline.py` -- `AudioPreprocessingPipeline`
- `src/theo/preprocessing/audio_io.py` -- funcoes de conversao bytes <-> numpy e deteccao de formato
- Teste: `tests/unit/test_preprocessing_pipeline.py`

**Detalhes de implementacao**:

```python
class AudioStage(ABC):
    """Stage do pipeline de preprocessamento de audio.

    Cada stage recebe um array numpy mono e retorna array processado.
    Stateless por design — nenhum estado acumulado entre chamadas.
    """

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Processa frame de audio.

        Args:
            audio: Array numpy float32 mono.
            sample_rate: Sample rate atual.

        Returns:
            Tupla (audio processado, sample rate apos processamento).
        """
        ...


class AudioPreprocessingPipeline:
    """Orquestra stages de preprocessamento de audio.

    Recebe audio bruto (bytes de um upload WAV/MP3/FLAC/etc),
    detecta formato e sample rate, aplica stages habilitados,
    e retorna PCM 16-bit 16kHz mono normalizado.
    """

    def __init__(self, config: PreprocessingConfig, stages: list[AudioStage] | None = None) -> None:
        ...

    def process(self, audio_bytes: bytes) -> bytes:
        """Processa audio completo (modo batch).

        Args:
            audio_bytes: Bytes brutos do arquivo de audio.

        Returns:
            PCM 16-bit 16kHz mono como bytes.

        Raises:
            AudioFormatError: Se o formato de audio nao puder ser decodificado.
        """
        ...
```

**Deteccao de formato e sample rate**: Usar `wave` (stdlib) para WAV. Para MP3/FLAC/OGG, avaliar dependencia minima. Opcoes:

| Formato | Opcao A (stdlib) | Opcao B (dependencia leve) |
|---------|-------------------|---------------------------|
| WAV | `wave` module (stdlib) | -- |
| MP3, FLAC, OGG | Nao suportado | `soundfile` (libsndfile wrapper, suporta WAV/FLAC/OGG) |

**Decisao**: Usar `soundfile` como dependencia porque o Theo ja aceita MP3/FLAC/OGG na API (M3 valida content-types para esses formatos). Se nao decodificarmos esses formatos, o preprocessing quebraria para uploads nao-WAV. `soundfile` e leve (~100KB, depende de `libsndfile` que ja e ubiqua em Linux).

Alternativa avaliada: `pydub` (depende de `ffmpeg`). Descartada porque `ffmpeg` e dependencia pesada de sistema. `soundfile` resolve sem dependencia externa adicional.

**Definition of Done**:
- Pipeline instancia com `PreprocessingConfig` default e lista de stages vazia
- `pipeline.process(wav_bytes)` com zero stages retorna audio decodificado como PCM 16-bit
- `AudioStage` ABC nao pode ser instanciada diretamente
- Teste com stage passthrough (identity) retorna audio inalterado
- Deteccao de sample rate funciona para WAV
- `AudioFormatError` levantada para bytes invalidos
- mypy e ruff passam

**Dependencias**: Nenhuma task de M4 (usa `PreprocessingConfig` de M1, `AudioFormatError` de M1)

**Estimativa**: M (1-2 dias)

---

#### E1-T2: Stage Resample

**Contexto/Motivacao**: O Faster-Whisper (e todas as engines STT) espera audio em 16kHz. Clientes enviam audio em qualquer sample rate: 44.1kHz (gravacao desktop), 48kHz (WebRTC), 8kHz (telefonia). Sem resample, o audio e interpretado com sample rate errado e a transcricao falha completamente ou produz lixo.

**Escopo**:

| In | Out |
|----|-----|
| `ResampleStage` que converte qualquer SR para target (default 16kHz) | Suporte a formatos nao-PCM (MP3 decode) |
| Usa `scipy.signal.resample_poly` como implementacao | soxr (opcional futuro, melhor qualidade) |
| Converte stereo para mono (mean dos canais) se necessario | |
| Skip se audio ja esta no target sample rate | |
| Teste com audio 44.1kHz e 8kHz | |

**Entregaveis**:
- `src/theo/preprocessing/resample.py` -- `ResampleStage`
- Teste: `tests/unit/test_preprocessing_resample.py`

**Detalhes de implementacao**:

```python
class ResampleStage(AudioStage):
    """Converte audio para target sample rate.

    Usa scipy.signal.resample_poly para resampling de alta qualidade.
    Tambem converte stereo para mono (mean dos canais) se necessario.
    """

    def __init__(self, target_sample_rate: int = 16000) -> None:
        self._target_sr = target_sample_rate

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        # Se ja esta no target, skip
        if sample_rate == self._target_sr:
            return audio, sample_rate

        # Converte stereo para mono se necessario
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample via scipy
        resampled = scipy.signal.resample_poly(audio, up=self._target_sr, down=sample_rate)
        return resampled.astype(np.float32), self._target_sr
```

**Por que `scipy.signal.resample_poly`?** `scipy` ja e dependencia transitiva do numpy. `resample_poly` usa filtro anti-aliasing FIR e e mais eficiente que `resample` para fatores inteiros. Qualidade superior a resampling linear/cubic.

**Alternativa avaliada**: `soxr-python` (wrapper para libsoxr, considerada a melhor qualidade de resampling disponivel). Nao adicionada agora porque `scipy` resolve e nao adiciona dependencia extra. Se testes de qualidade em M8 (telefonia) mostrarem degradacao, substituir por `soxr` e trivial gracas a interface `AudioStage`.

**Definition of Done**:
- Audio 44.1kHz mono resampleado para 16kHz: comprimento correto (proporcional ao ratio)
- Audio 48kHz stereo resampleado para 16kHz mono
- Audio 8kHz resampleado para 16kHz
- Audio 16kHz nao e alterado (skip)
- Teste com fixture WAV real (gerada em conftest)
- mypy e ruff passam

**Dependencias**: E1-T1 (interface AudioStage existe)

**Estimativa**: S (0.5-1 dia)

---

#### E1-T3: Stage DC Remove

**Contexto/Motivacao**: Audio de telefonia e hardware barato frequentemente contem DC offset -- um valor medio nao-zero no sinal que desperdiça headroom e pode confundir o VAD e a engine. Um high-pass filter simples a 20Hz remove o offset sem afetar a fala (banda util: 80Hz-8kHz).

**Escopo**:

| In | Out |
|----|-----|
| `DCRemoveStage` com HPF Butterworth 2a ordem, fc=20Hz | Filtros adaptativos |
| Pre-calcula coeficientes no `__init__` | |
| Usa `scipy.signal.sosfilt` para filtragem | |
| Teste com sinal contendo DC offset (0.1 adicionado) | |

**Entregaveis**:
- `src/theo/preprocessing/dc_remove.py` -- `DCRemoveStage`
- Teste: `tests/unit/test_preprocessing_dc_remove.py`

**Detalhes de implementacao**:

```python
class DCRemoveStage(AudioStage):
    """Remove DC offset via high-pass filter Butterworth.

    Aplica HPF de 2a ordem com frequencia de corte configuravel (default 20Hz).
    Coeficientes sao pre-calculados no init para evitar recomputacao.
    """

    def __init__(self, cutoff_hz: int = 20) -> None:
        self._cutoff_hz = cutoff_hz
        # Coeficientes calculados no primeiro process() porque dependem do sample_rate
        self._sos: np.ndarray | None = None
        self._last_sr: int = 0

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        if self._sos is None or sample_rate != self._last_sr:
            self._sos = scipy.signal.butter(
                2, self._cutoff_hz, btype="highpass", fs=sample_rate, output="sos"
            )
            self._last_sr = sample_rate

        filtered = scipy.signal.sosfilt(self._sos, audio).astype(np.float32)
        return filtered, sample_rate
```

**Por que coeficientes lazy?** Os coeficientes Butterworth dependem do sample rate. No pipeline batch, o sample rate e conhecido apos o resample (16kHz). Mas a interface AudioStage recebe `sample_rate` como parametro, entao calculamos na primeira chamada e cacheamos.

**Definition of Done**:
- Sinal com DC offset de 0.1 (media nao-zero): apos filtragem, media < 0.001
- Sinal puro (sem offset) nao e degradado significativamente (SNR > 40dB)
- Coeficientes sao recalculados se sample rate mudar entre chamadas
- mypy e ruff passam

**Dependencias**: E1-T1 (interface AudioStage existe)

**Estimativa**: S (0.5 dia)

---

#### E1-T4: Stage Gain Normalize

**Contexto/Motivacao**: Engines CTC sao especialmente sensiveis a amplitude do audio. Audio muito baixo (microfone distante) ou muito alto (clipping) degrada WER significativamente. Normalizar a amplitude para um target consistente (-3dBFS) garante que o audio chega a engine em condicoes otimas, independente da fonte.

**Escopo**:

| In | Out |
|----|-----|
| `GainNormalizeStage` com peak normalization para target dBFS | RMS normalization (alternativa) |
| Target configuravel (default -3.0 dBFS) | AGC (automatic gain control) |
| Clipping protection: limitar a 0dBFS apos aplicacao de ganho | |
| Skip se audio ja esta proximo do target (+-1dB) | |
| Teste com audio baixo (-20dBFS) e alto (-1dBFS) | |

**Entregaveis**:
- `src/theo/preprocessing/gain_normalize.py` -- `GainNormalizeStage`
- Teste: `tests/unit/test_preprocessing_gain_normalize.py`

**Detalhes de implementacao**:

```python
class GainNormalizeStage(AudioStage):
    """Normaliza amplitude do audio para target dBFS.

    Usa peak normalization: calcula pico do sinal e aplica ganho
    para que o pico atinja o target_dbfs.
    """

    def __init__(self, target_dbfs: float = -3.0) -> None:
        self._target_dbfs = target_dbfs
        self._target_linear = 10 ** (target_dbfs / 20)

    def process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            # Silencio — nao aplicar ganho (divisao por zero)
            return audio, sample_rate

        gain = self._target_linear / peak
        normalized = audio * gain

        # Clipping protection
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32), sample_rate
```

**Peak normalization vs RMS normalization**: O PRD especifica "peak normalization para -3dBFS". Peak e mais simples e previsiivel. RMS consideraria a energia media, mas e mais complexo e pode nao lidar bem com picos transientes. KISS.

**Definition of Done**:
- Audio com pico em -20dBFS: apos normalizacao, pico em -3dBFS (+- 0.1dB)
- Audio com pico em -1dBFS: apos normalizacao, pico em -3dBFS
- Audio silencioso (todo zeros): retornado inalterado, sem divisao por zero
- Audio que resultaria em clipping apos ganho: limitado a 0dBFS
- mypy e ruff passam

**Dependencias**: E1-T1 (interface AudioStage existe)

**Estimativa**: S (0.5 dia)

---

#### E1-T5: Integracao Preprocessing no Fluxo Batch

**Contexto/Motivacao**: Os stages individuais estao prontos. Agora precisamos conectar o pipeline ao fluxo existente de `handle_audio_request()`. Este e o momento em que um `curl -F file=@audio_44khz.wav` passa a funcionar corretamente.

**Escopo**:

| In | Out |
|----|-----|
| Instanciar `AudioPreprocessingPipeline` no `create_app()` | Configuracao por request |
| Injetar pipeline em `handle_audio_request()` via dependencia | Modo streaming |
| Aplicar preprocessing entre leitura do arquivo e criacao do `TranscribeRequest` | |
| `PreprocessingConfig` carregada no startup (defaults por agora) | |
| Teste e2e: audio 44.1kHz preprocessado e transcrito corretamente | |

**Entregaveis**:
- Alteracao em `src/theo/server/app.py` -- instancia pipeline, armazena em `app.state`
- Alteracao em `src/theo/server/dependencies.py` -- `get_preprocessing_pipeline()`
- Alteracao em `src/theo/server/routes/_common.py` -- aplica preprocessing no audio
- Teste: `tests/unit/test_preprocessing_integration.py`

**Fluxo apos integracao**:

```
handle_audio_request():
  1. Valida input (file, model, size, content-type)     [existente]
  2. audio_data = await file.read()                      [existente]
  3. processed_audio = pipeline.process(audio_data)      [NOVO]
  4. request = TranscribeRequest(audio_data=processed_audio, ...)  [existente, com audio processado]
  5. result = await scheduler.transcribe(request)        [existente]
  6. return format_response(result, fmt)                 [existente]
```

**Detalhes de integracao**:

O `AudioPreprocessingPipeline` recebe `bytes` (raw do upload) e retorna `bytes` (PCM 16-bit 16kHz mono). O `TranscribeRequest.audio_data` ja espera `bytes`. A integracao e transparente.

O pipeline e instanciado uma vez no startup (`create_app`) e compartilhado entre requests. Ele e stateless (cada `process()` e independente), entao compartilhar e seguro.

**Definition of Done**:
- Audio WAV 44.1kHz enviado via curl: worker recebe PCM 16kHz (verificar nos logs do worker)
- Audio WAV 16kHz: funciona como antes (pipeline faz skip no resample)
- Audio WAV 8kHz: resampleado para 16kHz, transcrito corretamente
- `create_app()` sem pipeline (backwards compatible para testes existentes): funciona sem quebrar
- Todos os testes existentes de M3 continuam passando
- mypy e ruff passam

**Dependencias**: E1-T2, E1-T3, E1-T4 (todos os stages implementados)

**Estimativa**: M (1-1.5 dias)

---

### Epic 2 -- Post-Processing Pipeline

#### E2-T1: Interface TextStage + PostProcessingPipeline

**Contexto/Motivacao**: Analogamente ao preprocessing, o post-processing precisa de um esqueleto antes dos stages. A interface e mais simples (texto -> texto), mas o padrao de orquestracao e o mesmo: lista de stages habilitados via config.

**Escopo**:

| In | Out |
|----|-----|
| ABC `TextStage` com metodo `process(text) -> text` | Entity Formatting (futuro) |
| `PostProcessingPipeline` com `process(text) -> text` | Hot Word Correction (futuro) |
| Pipeline aceita lista de stages habilitados via `PostProcessingConfig` | |
| Aplicacao em `BatchResult.text` E em cada `segment.text` | |
| Teste unitario com stage identity (passthrough) | |

**Entregaveis**:
- `src/theo/postprocessing/stages.py` -- ABC `TextStage`
- `src/theo/postprocessing/pipeline.py` -- `PostProcessingPipeline`
- Teste: `tests/unit/test_postprocessing_pipeline.py`

**Detalhes de implementacao**:

```python
class TextStage(ABC):
    """Stage do pipeline de pos-processamento de texto.

    Cada stage recebe texto e retorna texto transformado.
    Stateless por design.
    """

    @abstractmethod
    def process(self, text: str) -> str:
        """Processa texto.

        Args:
            text: Texto cru ou parcialmente processado.

        Returns:
            Texto apos processamento deste stage.
        """
        ...


class PostProcessingPipeline:
    """Orquestra stages de pos-processamento de texto.

    Aplica sequencialmente cada stage habilitado ao texto.
    Aplica tanto ao texto principal quanto a cada segmento individual.
    """

    def __init__(self, config: PostProcessingConfig, stages: list[TextStage] | None = None) -> None:
        ...

    def process(self, text: str) -> str:
        """Processa texto individual atraves de todos os stages."""
        ...

    def process_result(self, result: BatchResult) -> BatchResult:
        """Processa um BatchResult completo (texto principal + segmentos).

        Retorna novo BatchResult com textos processados.
        BatchResult e frozen dataclass -- cria nova instancia.
        """
        ...
```

**Nota sobre imutabilidade**: `BatchResult` e `SegmentDetail` sao `frozen=True`. O pipeline cria novas instancias em vez de mutar. Isso e consistente com o design existente.

**Definition of Done**:
- Pipeline instancia com `PostProcessingConfig` default e lista de stages vazia
- `pipeline.process("hello")` sem stages retorna `"hello"` inalterado
- `pipeline.process_result(batch_result)` sem stages retorna `BatchResult` com mesmos dados
- `TextStage` ABC nao pode ser instanciada diretamente
- mypy e ruff passam

**Dependencias**: Nenhuma task de M4 (usa `PostProcessingConfig` de M1, `BatchResult` de M1)

**Estimativa**: S (0.5-1 dia)

---

#### E2-T2: Stage ITN (nemo_text_processing)

**Contexto/Motivacao**: Inverse Text Normalization e o stage mais valioso do post-processing. Transforma texto verbalizado em formato escrito: "dois mil e vinte e cinco" -> "2025", "dez por cento" -> "10%". Sem ITN, o output do STT e ilegivel em contextos que envolvem numeros, datas, porcentagens e valores monetarios.

**Escopo**:

| In | Out |
|----|-----|
| `ITNStage` que usa `nemo_text_processing` para pt-BR | ITN para outros idiomas |
| Fallback graceful se `nemo_text_processing` nao instalado | Regras customizadas |
| Configuracao de idioma via `ITNConfig.language` | |
| Tratamento de erros: excecao do NeMo nao propaga, retorna texto original com log warning | |
| Lazy loading do modelo NeMo (heavy init) | |
| Teste unitario com mock de `nemo_text_processing` | |
| Teste de integracao com `nemo_text_processing` real (marcado integration) | |

**Entregaveis**:
- `src/theo/postprocessing/itn.py` -- `ITNStage`
- Teste: `tests/unit/test_postprocessing_itn.py`
- Teste: `tests/integration/test_itn_real.py` (marcado `@pytest.mark.integration`)

**Detalhes de implementacao**:

```python
class ITNStage(TextStage):
    """Inverse Text Normalization via nemo_text_processing.

    Converte texto verbalizado em formato escrito:
    - "dois mil e vinte e cinco" -> "2025"
    - "dez por cento" -> "10%"
    - "quinze reais e cinquenta centavos" -> "R$15,50"

    Se nemo_text_processing nao esta instalado, retorna texto
    inalterado com warning no log. Fail-open, nao fail-closed.
    """

    def __init__(self, language: str = "pt") -> None:
        self._language = language
        self._normalizer: Any | None = None
        self._available: bool | None = None  # None = nao verificado ainda

    def _ensure_loaded(self) -> bool:
        """Lazy load do normalizer NeMo. Retorna True se disponivel."""
        if self._available is not None:
            return self._available
        try:
            from nemo_text_processing.inverse_text_normalization import InverseNormalize
            self._normalizer = InverseNormalize(lang=self._language)
            self._available = True
        except ImportError:
            logger.warning("nemo_text_processing_not_available", ...)
            self._available = False
        except Exception:
            logger.warning("itn_init_failed", ...)
            self._available = False
        return self._available

    def process(self, text: str) -> str:
        if not text.strip():
            return text

        if not self._ensure_loaded():
            return text  # Fallback: retorna texto original

        try:
            return self._normalizer.inverse_normalize(text, verbose=False)
        except Exception:
            logger.warning("itn_process_failed", text_length=len(text))
            return text  # Fallback: retorna texto original
```

**Lazy loading**: O `InverseNormalize` do NeMo leva 1-3 segundos para inicializar (carrega gramatica WFST). Lazy loading garante que o custo so e pago na primeira transcricao, nao no startup do server.

**Fail-open**: Se o NeMo falhar (importacao, init, processamento), o stage retorna o texto original sem modificacao. A transcricao funciona -- so sem formatacao. Isso e melhor do que falhar a request inteira por causa de ITN.

**Definition of Done**:
- Teste unitario com mock: `ITNStage.process("dois mil")` chama mock do NeMo e retorna resultado
- Teste unitario: sem `nemo_text_processing` importavel, retorna texto original e loga warning
- Teste unitario: NeMo levanta excecao durante `inverse_normalize`, retorna texto original
- Teste unitario: texto vazio retorna texto vazio sem chamar NeMo
- Teste unitario: lazy loading so chama import na primeira chamada
- Teste de integracao (marcado `@pytest.mark.integration`): `"dois mil e vinte e cinco"` -> `"2025"` com NeMo real
- mypy e ruff passam

**Dependencias**: E2-T1 (interface TextStage existe)

**Estimativa**: M (1-1.5 dias)

---

#### E2-T3: Integracao Post-Processing no Fluxo Batch

**Contexto/Motivacao**: O stage ITN esta pronto. Agora precisamos conectar o pipeline ao fluxo existente de `handle_audio_request()`. O `BatchResult` da engine passa pelo post-processing antes de ser formatado.

**Escopo**:

| In | Out |
|----|-----|
| Instanciar `PostProcessingPipeline` no `create_app()` | Configuracao por request |
| Injetar pipeline em `handle_audio_request()` via dependencia | |
| Aplicar post-processing no `BatchResult` antes de `format_response()` | |
| `PostProcessingConfig` carregada no startup (defaults: ITN habilitado) | |
| Post-processing aplicado em `BatchResult.text` e em cada `segment.text` | |

**Entregaveis**:
- Alteracao em `src/theo/server/app.py` -- instancia pipeline, armazena em `app.state`
- Alteracao em `src/theo/server/dependencies.py` -- `get_postprocessing_pipeline()`
- Alteracao em `src/theo/server/routes/_common.py` -- aplica post-processing no resultado
- Teste: `tests/unit/test_postprocessing_integration.py`

**Fluxo apos integracao**:

```
handle_audio_request():
  1. Valida input                                        [existente]
  2. audio_data = await file.read()                      [existente]
  3. processed_audio = pre_pipeline.process(audio_data)  [de E1-T5]
  4. request = TranscribeRequest(audio_data=processed_audio, ...)
  5. result = await scheduler.transcribe(request)        [existente]
  6. result = post_pipeline.process_result(result)       [NOVO]
  7. return format_response(result, fmt)                 [existente]
```

**Definition of Done**:
- `BatchResult.text` com numeros verbalizados: apos post-processing, numeros formatados
- `BatchResult.segments[*].text` tambem processados
- `create_app()` sem pipeline (backwards compatible): funciona sem quebrar
- Todos os testes existentes de M3 continuam passando (pipeline com ITN desabilitado ou mockado)
- mypy e ruff passam

**Dependencias**: E2-T2 (stage ITN implementado), E1-T5 (preprocessing ja integrado)

**Estimativa**: S (0.5-1 dia)

---

### Epic 3 -- Controle e Validacao

#### E3-T1: CLI `--no-itn` + API `itn` param

**Contexto/Motivacao**: O usuario precisa de controle sobre o ITN. Em alguns cenarios (corpus para treinamento, debugging, texto cru para processamento downstream), o ITN e indesejado. O PRD define: campo `itn` no endpoint REST (default `true`), flag `--no-itn` no CLI.

**Escopo**:

| In | Out |
|----|-----|
| Campo `itn` no form-data de `/v1/audio/transcriptions` | Campo `itn` no WebSocket `session.configure` (M5) |
| Campo `itn` no form-data de `/v1/audio/translations` | |
| Flag `--no-itn` em `theo transcribe` | |
| Flag `--no-itn` em `theo translate` | |
| Quando `itn=false`, post-processing pipeline pula o stage ITN | |

**Entregaveis**:
- Alteracao em `src/theo/server/routes/transcriptions.py` -- campo `itn: bool = Form(default=True)`
- Alteracao em `src/theo/server/routes/translations.py` -- campo `itn: bool = Form(default=True)`
- Alteracao em `src/theo/server/routes/_common.py` -- propagar `itn` para controle do pipeline
- Alteracao em `src/theo/cli/transcribe.py` -- flag `--no-itn`
- Teste: `tests/unit/test_itn_control.py`

**Detalhes de implementacao**:

A forma mais simples de implementar o controle: o parametro `itn` e passado para `handle_audio_request()`, que decide se chama `post_pipeline.process_result()` ou nao. Alternativa mais granular: o pipeline recebe um override que desabilita stages especificos. Mas para M4, com apenas um stage, a abordagem simples (skip pipeline inteiro) e suficiente. KISS.

Quando Entity Formatting e Hot Word Correction forem adicionados (futuro), refatorar para controle por stage.

**Nota sobre compatibilidade OpenAI**: O campo `itn` NAO existe na API OpenAI. Campos extras em multipart/form-data sao ignorados pelo SDK OpenAI Python, entao nao quebra compatibilidade. O SDK nao envia `itn`, entao o default `True` se aplica.

**Definition of Done**:
- `curl ... -F itn=false` retorna texto sem formatacao ITN
- `curl ... -F itn=true` (ou sem campo) retorna texto com ITN
- `theo transcribe audio.wav --no-itn` retorna texto sem ITN
- `theo translate audio.wav --no-itn` retorna texto sem ITN
- Testes unitarios cobrindo ambos os cenarios
- mypy e ruff passam

**Dependencias**: E2-T3 (post-processing integrado no fluxo)

**Estimativa**: S (0.5 dia)

---

#### E3-T2: Testes End-to-End Pipeline Completo

**Contexto/Motivacao**: Os testes unitarios validam stages isolados e integracao individual. Os testes e2e validam o fluxo completo: audio em qualquer sample rate -> preprocessing -> worker (mock) -> post-processing -> resposta formatada. Sao a prova definitiva de que M4 funciona.

**Escopo**:

| In | Out |
|----|-----|
| Teste e2e com audio 44.1kHz -> preprocessing -> transcricao | Teste com modelo real |
| Teste e2e com audio 8kHz -> preprocessing -> transcricao | Teste com GPU |
| Teste e2e verificando ITN aplicado no texto final | |
| Teste e2e verificando `itn=false` retorna texto cru | |
| Teste e2e verificando que todos os formatos (srt, vtt, etc) contem texto pos-processado | |
| Teste e2e verificando que segments tambem sao pos-processados | |
| Mock do Scheduler (nao precisa de worker real) | |

**Entregaveis**:
- `tests/unit/test_e2e_pipelines.py` -- testes e2e com mocks, roda no CI
- Atualizacao de `tests/conftest.py` -- fixtures de audio em 44.1kHz e 8kHz (se nao existirem)

**Estrategia de mock**:

O teste cria mocks do Scheduler e do NeMo:
- `MockScheduler` retorna `BatchResult` com texto pre-definido (ex: `"dois mil e quinhentos reais"`)
- O `ITNStage` real e instanciado com mock do `nemo_text_processing` que transforma `"dois mil e quinhentos reais"` -> `"2.500 reais"`
- O pipeline de preprocessing e real (sem mock) -- testa conversao de sample rate de verdade

```python
async def test_full_pipeline_44khz_with_itn():
    """Audio 44.1kHz com ITN: preprocessing converte para 16kHz, ITN formata numeros."""
    # Arrange
    audio_44khz = generate_wav(sample_rate=44100, duration=1.0)
    mock_scheduler = MockScheduler(returns=BatchResult(text="dois mil e quinhentos reais", ...))
    mock_itn = MockITNStage(mapping={"dois mil e quinhentos reais": "2.500 reais"})

    app = create_app_with_pipelines(scheduler=mock_scheduler, itn_stage=mock_itn)
    async with AsyncClient(...) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", audio_44khz)},
            data={"model": "faster-whisper-tiny"},
        )

    # Assert
    assert response.status_code == 200
    assert response.json()["text"] == "2.500 reais"
    # Verificar que o scheduler recebeu audio com sample rate 16kHz
    received_audio = mock_scheduler.last_request.audio_data
    assert len(received_audio) < len(audio_44khz)  # menor porque 16kHz < 44.1kHz
```

**Definition of Done**:
- Teste com audio 44.1kHz: preprocessing funciona, transcricao retorna resultado
- Teste com audio 8kHz: preprocessing funciona, transcricao retorna resultado
- Teste com audio 16kHz: preprocessing faz skip no resample, resultado correto
- Teste com ITN habilitado: texto formatado no response
- Teste com ITN desabilitado (`itn=false`): texto cru no response
- Teste com formato `verbose_json`: segments tambem pos-processados
- Teste com formato `srt`/`vtt`: texto pos-processado nos subtitulos
- Todos os testes existentes de M3 continuam passando
- mypy e ruff passam

**Dependencias**: E3-T1 (controle de ITN funciona), E1-T5 (preprocessing integrado), E2-T3 (post-processing integrado)

**Estimativa**: M (1-1.5 dias)

---

## 4. Ordem de Execucao e Timeline

### Diagrama de Dependencias

```
E1-T1 (AudioStage + Pipeline)
  |
  +---> E1-T2 (Resample) ---+
  +---> E1-T3 (DC Remove) --+--> E1-T5 (Integracao pre-processing)
  +---> E1-T4 (Gain Norm) --+            |
                                          |
E2-T1 (TextStage + Pipeline)             |
  |                                       |
  +---> E2-T2 (ITN Stage) ------> E2-T3 (Integracao post-processing)
                                          |
                                   E3-T1 (CLI --no-itn + API itn)
                                          |
                                   E3-T2 (Testes e2e)
```

**Nota**: E1-T2, E1-T3 e E1-T4 podem ser implementados em paralelo porque dependem apenas de E1-T1 (a interface) e nao entre si. O mesmo vale para E2-T1 que pode comecar em paralelo com a Fase A.

### Timeline

```
Dia 1:
  E1-T1  Interface AudioStage + Pipeline              [1d]
  E2-T1  Interface TextStage + Pipeline (paralelo)     [0.5d inicio]

Dia 2:
  E2-T1  Interface TextStage + Pipeline (conclusao)    [0.5d]
  E1-T2  Stage Resample                                [0.5d]
  E1-T3  Stage DC Remove                               [0.5d]

Dia 3:
  E1-T4  Stage Gain Normalize                          [0.5d]
  E1-T5  Integracao preprocessing no fluxo batch       [0.5d inicio]

Dia 4:
  E1-T5  Integracao preprocessing (conclusao)          [0.5d]
  E2-T2  Stage ITN (nemo_text_processing)              [1d inicio]

  -- CHECKPOINT: audio de qualquer sample rate funciona --

Dia 5:
  E2-T2  Stage ITN (conclusao)                         [0.5d]
  E2-T3  Integracao post-processing no fluxo batch     [0.5d]

  -- CHECKPOINT: texto formatado (ITN) funciona --

Dia 6:
  E3-T1  CLI --no-itn + API itn param                 [0.5d]
  E3-T2  Testes e2e pipeline completo (inicio)         [0.5d]

Dia 7:
  E3-T2  Testes e2e (conclusao)                        [0.5d]
  Ajustes finais, review, CHANGELOG                    [0.5d]
```

**Total estimado: 7 dias de trabalho focado (~1.5 semanas)**

### Checkpoints de valor incremental

| Dia | O que funciona |
|-----|----------------|
| 1 | Interfaces AudioStage e TextStage definidas, pipelines instanciaveis |
| 2 | Audio 44.1kHz resampleado para 16kHz, DC offset removido |
| 3 | Audio normalizado para -3dBFS, pipeline completo com 3 stages |
| 4 | Preprocessing integrado no fluxo batch -- `curl` com audio 44kHz funciona |
| 5 | ITN funciona -- "dois mil" vira "2000" na resposta |
| 6 | Controle: `--no-itn` e `itn=false` funcionam |
| 7 | Todos os testes e2e passam, M4 completo |

---

## 5. Mudancas no pyproject.toml

As seguintes alteracoes sao necessarias no `pyproject.toml` para M4:

```toml
# Adicionar ao core dependencies
dependencies = [
    ...existing...,
    "scipy>=1.11,<2.0",             # NOVO: resample_poly, sosfilt (preprocessing)
    "soundfile>=0.12,<1.0",         # NOVO: decode WAV/FLAC/OGG (preprocessing)
]

# Adicionar mypy override para nemo_text_processing
[[tool.mypy.overrides]]
module = "nemo_text_processing.*"
ignore_missing_imports = true

# Adicionar mypy override para soundfile
[[tool.mypy.overrides]]
module = "soundfile.*"
ignore_missing_imports = true
```

**Justificativa**:
- `scipy`: Necessario para `resample_poly` (resample) e `sosfilt` (DC remove). Dependencia core porque preprocessing e parte fundamental do runtime. `scipy` e uma dependencia muito usada no ecossistema cientifico Python e adiciona ~30MB.
- `soundfile`: Necessario para decodificar formatos de audio (WAV, FLAC, OGG). A API ja aceita esses formatos (M3 valida content-types). Sem `soundfile`, o preprocessing nao consegue ler audio nao-WAV. Dependencia leve (~100KB, libsndfile ubiqua em Linux).
- `nemo_text_processing`: Ja esta declarado como extra opcional (`itn = ["nemo_text_processing>=1.1,<2.0"]`). Nao muda. O stage ITN faz import condicional.

---

## 6. Riscos e Mitigacoes

| # | Risco | Probabilidade | Impacto | Mitigacao |
|---|-------|--------------|---------|-----------|
| R1 | `nemo_text_processing` e uma dependencia pesada que puxa PyTorch | Alta | Medio | Ja e extra opcional (`pip install theo[itn]`). Stage ITN faz fallback graceful se nao instalado. Testes unitarios usam mock. |
| R2 | ITN introduz erros em edge cases (ex: "um" -> "1" quando e artigo) | Media | Medio | Testes com corpus pt-BR diversificado. Flag `--no-itn` permite desabilitar. Monitorar em producao. Fail-open: excecao do NeMo retorna texto original. |
| R3 | `soundfile` depende de `libsndfile` (biblioteca C) nao disponivel em todos os ambientes | Baixa | Medio | `libsndfile` esta disponivel em Ubuntu/Debian via `apt install libsndfile1`. Docker image base ja inclui. Para ambientes sem `libsndfile`, fallback para `wave` (stdlib) que suporta apenas WAV. |
| R4 | `scipy.signal.resample_poly` e lento para arquivos longos (>5 min) | Media | Baixo | Para batch (M4), audio de 5 min resampleado em <1s em CPU. Aceitavel. Para streaming (M5), o resample opera frame-by-frame (20ms) que e <1ms. |
| R5 | Preprocessing altera audio de forma que degrada WER da engine | Baixa | Alto | Gain normalize e DC remove sao operacoes padroes de DSP que melhoram (nao degradam) performance de STT. Resample de 44.1kHz->16kHz e necessario. Testes de integracao com modelo real validam. |
| R6 | Testes existentes de M3 quebram com novo pipeline | Media | Medio | Pipeline injetado via `app.state` com fallback para None. Testes existentes que nao passam pipeline continuam funcionando. Novos testes usam `create_app()` com pipeline. |
| R7 | `scipy` adiciona ~30MB ao tamanho de instalacao | Certa | Baixo | `scipy` e dependencia essencial para audio DSP de qualidade. Alternativa (`soxr-python`) e menor mas adiciona dependencia de build (Cython). `scipy` e mais pragmatico. |

---

## 7. Estrutura de Arquivos Final (M4)

```
src/theo/
  preprocessing/
    __init__.py
    stages.py              # ABC AudioStage                          [NOVO]
    pipeline.py            # AudioPreprocessingPipeline              [NOVO]
    audio_io.py            # Decode audio, deteccao de formato/SR    [NOVO]
    resample.py            # ResampleStage                           [NOVO]
    dc_remove.py           # DCRemoveStage                           [NOVO]
    gain_normalize.py      # GainNormalizeStage                      [NOVO]

  postprocessing/
    __init__.py
    stages.py              # ABC TextStage                           [NOVO]
    pipeline.py            # PostProcessingPipeline                  [NOVO]
    itn.py                 # ITNStage (nemo_text_processing)         [NOVO]

  server/
    app.py                 # create_app() atualizado com pipelines   [ALTERADO]
    dependencies.py        # get_preprocessing/postprocessing_pipeline [ALTERADO]
    routes/
      _common.py           # handle_audio_request() com pipelines    [ALTERADO]
      transcriptions.py    # campo itn: bool                         [ALTERADO]
      translations.py      # campo itn: bool                         [ALTERADO]

  cli/
    transcribe.py          # flag --no-itn                            [ALTERADO]

tests/
  unit/
    test_preprocessing_pipeline.py      # Pipeline + AudioStage ABC  [NOVO]
    test_preprocessing_resample.py      # ResampleStage              [NOVO]
    test_preprocessing_dc_remove.py     # DCRemoveStage              [NOVO]
    test_preprocessing_gain_normalize.py # GainNormalizeStage         [NOVO]
    test_preprocessing_integration.py   # Integracao no fluxo batch  [NOVO]
    test_postprocessing_pipeline.py     # Pipeline + TextStage ABC   [NOVO]
    test_postprocessing_itn.py          # ITNStage (com mock)        [NOVO]
    test_postprocessing_integration.py  # Integracao no fluxo batch  [NOVO]
    test_itn_control.py                 # --no-itn, itn=false        [NOVO]
    test_e2e_pipelines.py               # Testes e2e completos       [NOVO]
  integration/
    test_itn_real.py                    # ITN com NeMo real          [NOVO]
```

---

## 8. Criterio de Sucesso do M4

### 8.1 Funcional

| # | Criterio | Verificacao |
|---|----------|-------------|
| 1 | Audio WAV 44.1kHz transcrito corretamente | `curl -F file=@audio_44khz.wav -F model=...` retorna texto |
| 2 | Audio WAV 8kHz transcrito corretamente | `curl -F file=@audio_8khz.wav -F model=...` retorna texto |
| 3 | Audio WAV 16kHz funciona como antes | Nao regrediu |
| 4 | Texto com numeros formatado via ITN | "dois mil" no audio -> "2000" no response |
| 5 | `itn=false` retorna texto sem formatacao | `curl ... -F itn=false` retorna texto cru |
| 6 | `theo transcribe --no-itn` retorna texto cru | CLI funcional |
| 7 | Todos os formatos de resposta contem texto pos-processado | verbose_json, srt, vtt com ITN |
| 8 | Segments individuais tambem pos-processados | `segments[*].text` com ITN |
| 9 | Sem `nemo_text_processing` instalado, ITN desabilitado gracefully | Warning no log, texto cru retornado |
| 10 | Todos os testes M3 continuam passando | Nao regrediu |

### 8.2 Qualidade de Codigo

| # | Criterio | Comando |
|---|----------|---------|
| 1 | mypy strict sem erros | `.venv/bin/python -m mypy src/` |
| 2 | ruff check sem warnings | `.venv/bin/python -m ruff check src/ tests/` |
| 3 | ruff format sem diffs | `.venv/bin/python -m ruff format --check src/ tests/` |
| 4 | Todos os testes passam | `.venv/bin/python -m pytest tests/unit/ -v` |
| 5 | CI verde | GitHub Actions |

### 8.3 Testes (minimo)

| Tipo | Escopo | Quantidade minima |
|------|--------|-------------------|
| Unit | AudioStage ABC + pipeline (passthrough, ordem, config) | 5-7 |
| Unit | ResampleStage (44.1k, 48k, 8k, 16k skip, stereo->mono) | 5-7 |
| Unit | DCRemoveStage (offset removido, sinal limpo inalterado) | 3-5 |
| Unit | GainNormalizeStage (baixo, alto, silencio, clipping) | 4-6 |
| Unit | Integracao preprocessing no fluxo batch | 3-5 |
| Unit | TextStage ABC + pipeline (passthrough, ordem, config) | 4-6 |
| Unit | ITNStage (mock, fallback, erro, vazio, lazy load) | 6-8 |
| Unit | Integracao post-processing no fluxo batch | 3-5 |
| Unit | Controle ITN (itn=true/false, --no-itn) | 4-6 |
| E2E | Pipeline completo (pre + post, formatos, itn on/off) | 7-10 |
| Integration | ITN com NeMo real | 2-3 |
| **Total** | | **~45-65 novos testes** |

### 8.4 Documentacao

| # | Criterio |
|---|----------|
| 1 | CHANGELOG.md atualizado com entradas de M4 |
| 2 | CLAUDE.md atualizado (memory section com estado M4) |
| 3 | Docstrings em todas as interfaces publicas (AudioStage, TextStage, pipelines) |
| 4 | ROADMAP.md atualizado com status de M4 |

### 8.5 Demo Goal

Ao final de M4, a seguinte sequencia funciona sem erros:

```bash
# 1. Iniciar servidor
theo serve &

# 2. Transcrever audio 44.1kHz (antes falharia ou produziria lixo)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio_44khz.wav \
  -F model=faster-whisper-tiny
# -> {"text": "2.500 reais"}
# (audio original: voz dizendo "dois mil e quinhentos reais" em 44.1kHz)

# 3. Transcrever audio 8kHz (telefonia simulada)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio_8khz.wav \
  -F model=faster-whisper-tiny
# -> {"text": "o valor e 10%"}

# 4. Transcrever sem ITN
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio_44khz.wav \
  -F model=faster-whisper-tiny \
  -F itn=false
# -> {"text": "dois mil e quinhentos reais"}

# 5. CLI com ITN (default)
theo transcribe audio_44khz.wav --model faster-whisper-tiny
# -> 2.500 reais

# 6. CLI sem ITN
theo transcribe audio_44khz.wav --model faster-whisper-tiny --no-itn
# -> dois mil e quinhentos reais

# 7. Formato verbose_json com segments pos-processados
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio_44khz.wav \
  -F model=faster-whisper-tiny \
  -F response_format=verbose_json
# -> {"text": "2.500 reais", "segments": [{"text": "2.500 reais", ...}], ...}

# 8. Formato SRT com texto pos-processado
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio_44khz.wav \
  -F model=faster-whisper-tiny \
  -F response_format=srt
# -> 1
# -> 00:00:00,000 --> 00:00:02,500
# -> 2.500 reais

# 9. Verificar que testes antigos nao quebraram
.venv/bin/python -m pytest tests/unit/ -v
# -> todos passam (M1 + M2 + M3 + M4)
```

### 8.6 Checklist Final

```bash
# 1. Qualidade de codigo
.venv/bin/python -m ruff format --check src/ tests/
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m mypy src/

# 2. Testes unitarios
.venv/bin/python -m pytest tests/unit/ -v

# 3. Validacao de imports
python -c "
from theo.preprocessing.pipeline import AudioPreprocessingPipeline
from theo.preprocessing.stages import AudioStage
from theo.preprocessing.resample import ResampleStage
from theo.preprocessing.dc_remove import DCRemoveStage
from theo.preprocessing.gain_normalize import GainNormalizeStage
from theo.preprocessing.audio_io import decode_audio
from theo.postprocessing.pipeline import PostProcessingPipeline
from theo.postprocessing.stages import TextStage
from theo.postprocessing.itn import ITNStage
print('Todos os modulos M4 importaveis com sucesso')
"

# 4. Verificar que testes M3 nao quebraram
.venv/bin/python -m pytest tests/unit/test_e2e_batch.py -v
.venv/bin/python -m pytest tests/unit/test_formatters.py -v
.venv/bin/python -m pytest tests/unit/test_transcriptions_route.py -v
```

---

## 9. Transicao M4 -> M5

Ao completar M4, o time deve ter:

1. **Pipelines funcionais** -- Preprocessing normaliza audio de qualquer fonte. Post-processing formata texto via ITN. Ambos integrados ao fluxo batch.

2. **Interfaces estaveis** -- `AudioStage` e `TextStage` sao as interfaces que todos os stages futuros seguem. Nao devem mudar em M5+.

3. **Preprocessing pronto para streaming** -- A interface `AudioStage.process(audio, sample_rate)` recebe frames individuais. Em M5, o WebSocket handler chama os mesmos stages frame-by-frame em vez de processar o arquivo completo. A unica mudanca e no `AudioPreprocessingPipeline` que ganha um metodo `process_frame()` alem do `process()` batch.

4. **Post-processing com ponto de extensao claro** -- Em M6, adicionar Entity Formatting e Hot Word Correction e questao de implementar `TextStage` e registrar no pipeline. Zero mudanca no orquestrador.

5. **Pontos de extensao para M5**:
   - **Preprocessing streaming**: Adicionar `process_frame(frame: np.ndarray, sample_rate: int) -> np.ndarray` ao pipeline. Os stages individuais ja aceitam frames -- o pipeline so precisa expor isso.
   - **Post-processing em partials vs finals**: Em M5/M6, post-processing so e aplicado em `transcript.final`, nunca em `transcript.partial`. O controle de quando aplicar pertence ao Session Manager, nao ao pipeline.
   - **Stage Denoise**: Adicionar `DenoiseStage` implementando `AudioStage`, usando RNNoise. Habilitado por default para telefonia (M8). O pipeline ja suporta stages adicionais via config toggle.

**O primeiro commit de M5 sera**: adicionar endpoint WebSocket `WS /v1/realtime` com handshake, reaproveitando `AudioPreprocessingPipeline` para processar frames de audio do WebSocket frame-by-frame.

---

*Documento gerado por Sofia Castellani (Principal Solution Architect, ARCH). Sera atualizado conforme a implementacao do M4 avanca.*
