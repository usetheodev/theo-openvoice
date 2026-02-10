'use client';

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useSWR, { mutate } from 'swr';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import { Slider } from '../components/ui/slider';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import {
  cancelJob,
  createJob,
  createRealtimeConnection,
  fetchJobs,
  fetchModels,
  fetchQueue,
  synthesizeSpeech,
  type DemoJob,
  type DemoModel,
  type QueueMetrics,
} from '../lib/api';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const PRIORITY_OPTIONS = [
  { value: 'REALTIME', label: 'Realtime' },
  { value: 'BATCH', label: 'Batch' },
];

function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat('pt-BR', {
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(new Date(timestamp * 1000));
}

function statusVariant(status: string) {
  switch (status) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'destructive';
    case 'cancelled':
      return 'secondary';
    default:
      return 'outline';
  }
}

// ---------------------------------------------------------------------------
// Icons (inline SVG to avoid adding lucide-react dependency issues)
// ---------------------------------------------------------------------------

function MicIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" x2="12" y1="19" y2="22" />
    </svg>
  );
}

function StopIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}

function VolumeIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Hero Section
// ---------------------------------------------------------------------------

function HeroSection({ modelCount }: { modelCount: number }) {
  return (
    <section className="relative overflow-hidden rounded-2xl border bg-card px-8 py-12">
      {/* Background decoration */}
      <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/5 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-primary/5 blur-3xl" />

      <div className="relative z-10 flex flex-col items-center text-center">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <VolumeIcon className="h-5 w-5" />
          </div>
          <span className="text-sm font-medium tracking-wider text-muted-foreground uppercase">
            Theo OpenVoice
          </span>
        </div>

        <h1 className="gradient-text text-4xl font-bold tracking-tight sm:text-5xl">
          Runtime Unificado de Voz
        </h1>

        <p className="mt-4 max-w-2xl text-lg text-muted-foreground">
          STT + TTS em um unico binario. API compativel com OpenAI, streaming WebSocket,
          VAD inteligente, session management e recovery de falhas.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            {modelCount} modelo{modelCount !== 1 ? 's' : ''} instalado{modelCount !== 1 ? 's' : ''}
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            OpenAI-Compatible API
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            Full-Duplex STT + TTS
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            gRPC Worker Isolation
          </Badge>
        </div>

        <div className="mt-6 grid grid-cols-3 gap-8 text-center">
          <div>
            <p className="text-2xl font-bold">M1-M9</p>
            <p className="text-xs text-muted-foreground">Milestones completos</p>
          </div>
          <div>
            <p className="text-2xl font-bold">1600+</p>
            <p className="text-xs text-muted-foreground">Testes passando</p>
          </div>
          <div>
            <p className="text-2xl font-bold">&lt;300ms</p>
            <p className="text-xs text-muted-foreground">Target TTFB</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Dashboard Tab
// ---------------------------------------------------------------------------

interface DashboardTabProps {
  models: DemoModel[] | undefined;
  queue: QueueMetrics | undefined;
  jobs: DemoJob[] | undefined;
  defaultModel: string;
}

function DashboardTab({ models, queue, jobs, defaultModel }: DashboardTabProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [priority, setPriority] = useState<string>('REALTIME');
  const [language, setLanguage] = useState<string>('');
  const [submitState, setSubmitState] = useState<{
    isSubmitting: boolean;
    error?: string;
    success?: string;
  }>({ isSubmitting: false });

  const effectiveModel = selectedModel || defaultModel;

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setSelectedFile(file ?? null);
  }

  async function onSubmit() {
    if (!selectedFile) {
      setSubmitState({ isSubmitting: false, error: 'Selecione um arquivo de audio.' });
      return;
    }
    if (!effectiveModel) {
      setSubmitState({ isSubmitting: false, error: 'Nenhum modelo disponivel.' });
      return;
    }

    setSubmitState({ isSubmitting: true });
    try {
      await createJob({
        file: selectedFile,
        model: effectiveModel,
        priority,
        language: language || undefined,
      });
      setSubmitState({ isSubmitting: false, success: 'Job enviado com sucesso.' });
      setSelectedFile(null);
      mutate('jobs');
      mutate('queue');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Erro desconhecido';
      setSubmitState({ isSubmitting: false, error: message });
    }
  }

  async function onCancel(requestId: string) {
    try {
      await cancelJob(requestId);
      mutate('jobs');
      mutate('queue');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Erro ao cancelar';
      setSubmitState((prev) => ({ ...prev, error: message }));
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        {/* Models card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">M</span>
              Modelos Instalados
            </CardTitle>
            <CardDescription>
              {models?.length ? `${models.length} modelo(s) disponivel(is)` : 'Carregando modelos...'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {models && models.length > 0 ? (
              models.map((model) => (
                <div key={model.name} className="rounded-lg border p-3 transition-colors hover:bg-muted/50">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold">{model.name}</p>
                    <Badge variant="outline">{model.engine}</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{model.description}</p>
                  <div className="mt-1.5 flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">v{model.version}</Badge>
                    <Badge variant="secondary" className="text-xs">{model.type.toUpperCase()}</Badge>
                    {model.capabilities?.architecture && (
                      <Badge variant="secondary" className="text-xs">{String(model.capabilities.architecture)}</Badge>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">Nenhum modelo carregado.</p>
            )}
          </CardContent>
        </Card>

        {/* Queue card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">Q</span>
              Fila do Scheduler
            </CardTitle>
            <CardDescription>Profundidade da fila e prioridades em tempo real.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between rounded-lg bg-muted/50 p-4">
              <span className="text-sm text-muted-foreground">Requests pendentes</span>
              <span className="text-3xl font-bold">{queue?.depth ?? '—'}</span>
            </div>
            <div className="space-y-2">
              {queue ? (
                Object.entries(queue.depth_by_priority).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{key}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">Coletando metricas...</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* New Transcription form */}
      <Card>
        <CardHeader>
          <CardTitle>Nova Transcricao</CardTitle>
          <CardDescription>Envie um arquivo WAV/MP3 e escolha prioridade.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="file">Arquivo de audio</Label>
              <Input id="file" type="file" accept="audio/*" onChange={onFileChange} />
            </div>
            <div className="space-y-2">
              <Label>Modelo</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Selecione um modelo" />
                </SelectTrigger>
                <SelectContent>
                  {models?.filter((m) => m.type === 'stt').map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Prioridade</Label>
              <Select value={priority} onValueChange={setPriority}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRIORITY_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="language">Idioma (opcional)</Label>
              <Input
                id="language"
                placeholder="pt"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
              />
            </div>
          </div>
          {submitState.error && (
            <p className="text-sm text-destructive">{submitState.error}</p>
          )}
          {submitState.success && (
            <p className="text-sm text-emerald-600">{submitState.success}</p>
          )}
          <Button onClick={onSubmit} disabled={submitState.isSubmitting || !selectedFile}>
            {submitState.isSubmitting ? 'Enviando...' : 'Enviar transcricao'}
          </Button>
        </CardContent>
      </Card>

      {/* Jobs table */}
      <Card>
        <CardHeader>
          <CardTitle>Jobs Recentes</CardTitle>
          <CardDescription>Acompanhe status e resultados das transcricoes.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Modelo</TableHead>
                <TableHead>Prioridade</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Atualizado</TableHead>
                <TableHead>Resultado</TableHead>
                <TableHead className="text-right">Acoes</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs && jobs.length > 0 ? (
                jobs.map((job) => (
                  <TableRow key={job.request_id}>
                    <TableCell className="font-mono text-xs">{job.request_id.slice(0, 8)}</TableCell>
                    <TableCell>{job.model}</TableCell>
                    <TableCell>{job.priority}</TableCell>
                    <TableCell>
                      <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(job.updated_at)}
                    </TableCell>
                    <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
                      {job.result?.text ?? job.error ?? 'Aguardando...'}
                    </TableCell>
                    <TableCell className="text-right">
                      {job.status === 'queued' && (
                        <Button variant="ghost" size="sm" onClick={() => onCancel(job.request_id)}>
                          Cancelar
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={7} className="py-6 text-center text-sm text-muted-foreground">
                    Nenhum job enviado ainda.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Streaming STT Tab
// ---------------------------------------------------------------------------

interface TranscriptEntry {
  id: number;
  text: string;
  isFinal: boolean;
  timestamp: number;
}

function StreamingSTTTab({ models }: { models: DemoModel[] | undefined }) {
  const sttModels = useMemo(() => models?.filter((m) => m.type === 'stt') ?? [], [models]);
  const [selectedModel, setSelectedModel] = useState('');
  const [language, setLanguage] = useState('pt');
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [vadActive, setVadActive] = useState(false);
  const [currentPartial, setCurrentPartial] = useState('');
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptsEndRef = useRef<HTMLDivElement>(null);

  const effectiveModel = selectedModel || sttModels[0]?.name || '';

  // Auto-scroll transcript
  useEffect(() => {
    transcriptsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts, currentPartial]);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;

    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close();
    }
    audioContextRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'session.close' }));
      wsRef.current.close();
    }
    wsRef.current = null;

    setIsRecording(false);
    setIsConnected(false);
    setVadActive(false);
    setCurrentPartial('');
  }, []);

  async function startRecording() {
    if (!effectiveModel) {
      setError('Selecione um modelo STT.');
      return;
    }

    setError(null);
    setTranscripts([]);
    setCurrentPartial('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const ws = createRealtimeConnection(effectiveModel, language || undefined, {
        onSessionCreated: () => {
          setIsConnected(true);
        },
        onVadSpeechStart: () => {
          setVadActive(true);
        },
        onVadSpeechEnd: () => {
          setVadActive(false);
        },
        onTranscriptPartial: (text) => {
          setCurrentPartial(text);
        },
        onTranscriptFinal: (text, segmentId) => {
          setCurrentPartial('');
          setTranscripts((prev) => [
            ...prev,
            { id: segmentId, text, isFinal: true, timestamp: Date.now() },
          ]);
        },
        onError: (message) => {
          setError(message);
        },
        onClose: () => {
          stopRecording();
        },
      });
      wsRef.current = ws;

      // Wait for WebSocket to open before starting audio
      ws.onopen = () => {
        const audioContext = new AudioContext({ sampleRate: 16000 });
        audioContextRef.current = audioContext;

        const source = audioContext.createMediaStreamSource(stream);
        // Buffer size of 4096 at 16kHz = ~256ms frames
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (event) => {
          if (ws.readyState !== WebSocket.OPEN) return;

          const inputData = event.inputBuffer.getChannelData(0);
          // Convert float32 -> int16 PCM
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(pcm16.buffer);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
        setIsRecording(true);
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro ao acessar microfone';
      setError(message);
      stopRecording();
    }
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MicIcon className="h-5 w-5" />
            Streaming STT
          </CardTitle>
          <CardDescription>
            Transcricao em tempo real via WebSocket com VAD e partial transcripts.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Modelo STT</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Selecione modelo" />
                </SelectTrigger>
                <SelectContent>
                  {sttModels.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="stt-lang">Idioma</Label>
              <Input
                id="stt-lang"
                placeholder="pt"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={toggleRecording}
                variant={isRecording ? 'destructive' : 'default'}
                className="w-full gap-2"
              >
                {isRecording ? (
                  <>
                    <StopIcon className="h-4 w-4" />
                    Parar gravacao
                  </>
                ) : (
                  <>
                    <MicIcon className="h-4 w-4" />
                    Iniciar gravacao
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Status indicators */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`h-2.5 w-2.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-muted-foreground/30'}`} />
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className={`h-2.5 w-2.5 rounded-full ${vadActive ? 'bg-red-500' : 'bg-muted-foreground/30'}`} />
                {vadActive && (
                  <div className="animate-pulse-ring absolute inset-0 h-2.5 w-2.5 rounded-full bg-red-500" />
                )}
              </div>
              <span className="text-xs text-muted-foreground">
                {vadActive ? 'Fala detectada' : 'Silencio'}
              </span>
            </div>
            {isRecording && (
              <div className="flex items-center gap-1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="animate-waveform w-1 rounded-full bg-primary"
                    style={{ animationDelay: `${i * 0.15}s`, height: '4px' }}
                  />
                ))}
              </div>
            )}
          </div>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </CardContent>
      </Card>

      {/* Transcript output */}
      <Card>
        <CardHeader>
          <CardTitle>Transcricao</CardTitle>
          <CardDescription>
            {transcripts.length} segmento{transcripts.length !== 1 ? 's' : ''} confirmado{transcripts.length !== 1 ? 's' : ''}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-80 min-h-[120px] space-y-2 overflow-y-auto rounded-lg bg-muted/30 p-4">
            {transcripts.length === 0 && !currentPartial && (
              <p className="text-center text-sm text-muted-foreground">
                {isRecording
                  ? 'Aguardando fala...'
                  : 'Clique em "Iniciar gravacao" para comecar.'}
              </p>
            )}

            {transcripts.map((entry) => (
              <div key={`${entry.id}-${entry.timestamp}`} className="flex items-start gap-2">
                <Badge variant="success" className="mt-0.5 shrink-0 text-xs">
                  Final
                </Badge>
                <p className="text-sm">{entry.text}</p>
              </div>
            ))}

            {currentPartial && (
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="mt-0.5 shrink-0 text-xs">
                  Partial
                </Badge>
                <p className="text-sm italic text-muted-foreground">{currentPartial}</p>
              </div>
            )}

            <div ref={transcriptsEndRef} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TTS Playback Tab
// ---------------------------------------------------------------------------

function TTSPlaybackTab({ models }: { models: DemoModel[] | undefined }) {
  const ttsModels = useMemo(() => models?.filter((m) => m.type === 'tts') ?? [], [models]);
  const [selectedModel, setSelectedModel] = useState('');
  const [text, setText] = useState('');
  const [voice, setVoice] = useState('default');
  const [speed, setSpeed] = useState(1.0);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ttfbMs, setTtfbMs] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  const effectiveModel = selectedModel || ttsModels[0]?.name || '';

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function onSynthesize() {
    if (!text.trim()) {
      setError('Digite um texto para sintetizar.');
      return;
    }
    if (!effectiveModel) {
      setError('Nenhum modelo TTS disponivel.');
      return;
    }

    setError(null);
    setIsSynthesizing(true);
    setTtfbMs(null);

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }

    const startTime = performance.now();

    try {
      const blob = await synthesizeSpeech({
        model: effectiveModel,
        input: text.trim(),
        voice,
        speed,
      });

      const elapsed = Math.round(performance.now() - startTime);
      setTtfbMs(elapsed);

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Auto-play
      setTimeout(() => {
        audioRef.current?.play();
      }, 50);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erro na sintese';
      setError(message);
    } finally {
      setIsSynthesizing(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <VolumeIcon className="h-5 w-5" />
            Text-to-Speech
          </CardTitle>
          <CardDescription>
            Sintetize voz a partir de texto via POST /v1/audio/speech (OpenAI-compatible).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="tts-text">Texto</Label>
            <textarea
              id="tts-text"
              className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              placeholder="Digite o texto que deseja sintetizar..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Modelo TTS</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Selecione modelo" />
                </SelectTrigger>
                <SelectContent>
                  {ttsModels.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                  {ttsModels.length === 0 && (
                    <SelectItem value="kokoro-v1" disabled>
                      Nenhum modelo TTS instalado
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="tts-voice">Voz</Label>
              <Input
                id="tts-voice"
                placeholder="default"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Velocidade: {speed.toFixed(1)}x</Label>
              <Slider
                min={0.5}
                max={2.0}
                step={0.1}
                value={[speed]}
                onValueChange={([v]) => setSpeed(v)}
              />
            </div>
          </div>

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}

          <div className="flex items-center gap-4">
            <Button
              onClick={onSynthesize}
              disabled={isSynthesizing || !text.trim()}
              className="gap-2"
            >
              <VolumeIcon className="h-4 w-4" />
              {isSynthesizing ? 'Sintetizando...' : 'Sintetizar voz'}
            </Button>

            {ttfbMs !== null && (
              <Badge variant="outline" className="text-xs">
                TTFB: {ttfbMs}ms
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Audio player */}
      {audioUrl && (
        <Card>
          <CardHeader>
            <CardTitle>Resultado</CardTitle>
            <CardDescription>Audio sintetizado — clique play para ouvir.</CardDescription>
          </CardHeader>
          <CardContent>
            <audio
              ref={audioRef}
              controls
              src={audioUrl}
              className="w-full"
            />
          </CardContent>
        </Card>
      )}

      {/* Architecture info */}
      <Card className="border-dashed">
        <CardContent className="py-6">
          <div className="grid gap-4 text-sm md:grid-cols-3">
            <div>
              <p className="font-semibold">Endpoint</p>
              <p className="text-muted-foreground">POST /v1/audio/speech</p>
            </div>
            <div>
              <p className="font-semibold">Formato</p>
              <p className="text-muted-foreground">WAV (PCM 16-bit) ou raw PCM</p>
            </div>
            <div>
              <p className="font-semibold">Streaming (WebSocket)</p>
              <p className="text-muted-foreground">tts.speak / tts.cancel com mute-on-speak</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function Page() {
  const { data: models } = useSWR<DemoModel[]>('models', fetchModels);
  const { data: queue } = useSWR<QueueMetrics>('queue', fetchQueue, {
    refreshInterval: 3000,
  });
  const { data: jobs } = useSWR<DemoJob[]>('jobs', fetchJobs, {
    refreshInterval: 4000,
  });

  const defaultModel = useMemo(() => {
    const sttModel = models?.find((m) => m.type === 'stt');
    return sttModel?.name ?? models?.[0]?.name ?? '';
  }, [models]);

  return (
    <main className="min-h-screen bg-background">
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 py-10">
        <HeroSection modelCount={models?.length ?? 0} />

        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard" className="gap-2">
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="streaming" className="gap-2">
              <MicIcon className="h-4 w-4" />
              Streaming STT
            </TabsTrigger>
            <TabsTrigger value="tts" className="gap-2">
              <VolumeIcon className="h-4 w-4" />
              Text-to-Speech
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard">
            <DashboardTab
              models={models}
              queue={queue}
              jobs={jobs}
              defaultModel={defaultModel}
            />
          </TabsContent>

          <TabsContent value="streaming">
            <StreamingSTTTab models={models} />
          </TabsContent>

          <TabsContent value="tts">
            <TTSPlaybackTab models={models} />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <footer className="border-t pt-6 text-center text-xs text-muted-foreground">
          <p>
            Theo OpenVoice — Runtime unificado de voz (STT + TTS).
            Todas as 3 fases do PRD entregues (M1-M9). 1600+ testes.
          </p>
          <p className="mt-1">
            API compativel com OpenAI · gRPC worker isolation · Full-duplex WebSocket · Mute-on-speak
          </p>
        </footer>
      </section>
    </main>
  );
}
