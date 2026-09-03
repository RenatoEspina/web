"use client";

import {
  ArrowUp,
  Bot,
  CircleAlert,
  CircleCheck,
  FileText,
  FileUp,
  FlaskConical,
  KeyRound,
  LoaderCircle,
  Plus,
  ShieldCheck,
  Sparkles,
  Trash2,
  WifiOff,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { DocumentSummary, KnowledgeMode, KnowledgeSource } from "@/lib/documents";

type UiMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: KnowledgeSource[];
  mode?: KnowledgeMode;
  cacheHit?: boolean;
  embeddingUsed?: boolean;
  contextTruncated?: boolean;
};

type GatewayConfig = {
  provider: "vllm" | "ollama";
  model: string;
  authRequired: boolean;
  models: string[];
  embedding: {
    enabled: boolean;
    provider: "vllm" | "ollama";
    model: string;
  };
};

type ConnectionStatus = "checking" | "online" | "offline" | "locked";

const TOKEN_STORAGE_KEY = "llm-bridge.app-token";
const WORKSPACE_STORAGE_KEY = "llm-bridge.workspace-id";

function authHeaders(token: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function workspaceHeaders(workspaceId: string): Record<string, string> {
  return workspaceId ? { "x-workspace-id": workspaceId } : {};
}

function getWorkspaceId(): string {
  if (typeof window === "undefined") return "";

  const existing = window.localStorage.getItem(WORKSPACE_STORAGE_KEY);
  if (existing && /^[A-Za-z0-9_-]{16,80}$/.test(existing)) return existing;

  const next = crypto.randomUUID();
  window.localStorage.setItem(WORKSPACE_STORAGE_KEY, next);
  return next;
}

function parseSources(value: unknown): KnowledgeSource[] {
  if (!Array.isArray(value)) return [];

  return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .filter((item) =>
      typeof item.documentId === "string" &&
      typeof item.documentName === "string" &&
      typeof item.page === "number" &&
      typeof item.chunkId === "string" &&
      typeof item.snippet === "string",
    )
    .map((item) => ({
      documentId: item.documentId as string,
      documentName: item.documentName as string,
      page: item.page as number,
      ...(typeof item.pageEnd === "number" ? { pageEnd: item.pageEnd as number } : {}),
      chunkId: item.chunkId as string,
      snippet: item.snippet as string,
      ...(typeof item.score === "number" ? { score: item.score } : {}),
      ...(typeof item.lexicalScore === "number" ? { lexicalScore: item.lexicalScore } : {}),
      ...(typeof item.semanticScore === "number" ? { semanticScore: item.semanticScore } : {}),
    }));
}

async function responseData(response: Response): Promise<Record<string, unknown>> {
  try {
    const value: unknown = await response.json();
    return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

async function getConnectionStatus(token: string): Promise<ConnectionStatus> {
  try {
    const response = await fetch("/api/health", {
      headers: authHeaders(token),
      cache: "no-store",
    });
    if (response.status === 401) return "locked";
    return response.ok ? "online" : "offline";
  } catch {
    return "offline";
  }
}

export default function Home() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [config, setConfig] = useState<GatewayConfig | null>(null);
  const [workspaceId, setWorkspaceId] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [knowledgeMode, setKnowledgeMode] = useState<KnowledgeMode>("rag");
  const [selectedModel, setSelectedModel] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [tokenDraft, setTokenDraft] = useState("");
  const [showTokenDialog, setShowTokenDialog] = useState(false);
  const [tokenError, setTokenError] = useState("");
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function loadDocuments(token: string, currentWorkspaceId: string) {
    if (!currentWorkspaceId) return;

    try {
      const response = await fetch("/api/documents", {
        headers: {
          ...authHeaders(token),
          ...workspaceHeaders(currentWorkspaceId),
        },
        cache: "no-store",
      });
      const data = await responseData(response);
      if (!response.ok) {
        if (response.status !== 401) {
          setDocumentError(typeof data.error === "string" ? data.error : "No fue posible cargar los documentos.");
        }
        return;
      }

      const nextDocuments = Array.isArray(data.documents)
        ? data.documents.filter((item): item is DocumentSummary => Boolean(item) && typeof item === "object" && typeof (item as Record<string, unknown>).id === "string")
        : [];
      setDocuments(nextDocuments);
      setSelectedDocumentIds((current) => current.filter((id) => nextDocuments.some((document) => document.id === id)));
    } catch {
      setDocumentError("No fue posible cargar la biblioteca de documentos.");
    }
  }

  useEffect(() => {
    let active = true;
    const storedToken = window.sessionStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
    const currentWorkspaceId = getWorkspaceId();

    async function bootstrap() {
      try {
        const response = await fetch("/api/config", { cache: "no-store" });
        const data = await responseData(response);
        if (!active) return;
        setWorkspaceId(currentWorkspaceId);
        setAuthToken(storedToken);
        setTokenDraft(storedToken);

        if (!response.ok || (data.provider !== "vllm" && data.provider !== "ollama")) {
          setStatus("offline");
          setError("La configuración del gateway no está disponible.");
          return;
        }

        const nextConfig: GatewayConfig = {
          provider: data.provider,
          model: typeof data.model === "string" ? data.model : "modelo configurado",
          authRequired: data.authRequired === true,
          models: Array.isArray(data.models) ? data.models.filter((model): model is string => typeof model === "string") : [typeof data.model === "string" ? data.model : "modelo configurado"],
          embedding: data.embedding && typeof data.embedding === "object"
            ? {
              enabled: (data.embedding as Record<string, unknown>).enabled !== false,
              provider: (data.embedding as Record<string, unknown>).provider === "vllm" ? "vllm" : "ollama",
              model: typeof (data.embedding as Record<string, unknown>).model === "string"
                ? (data.embedding as Record<string, unknown>).model as string
                : "modelo de embeddings",
            }
            : { enabled: true, provider: "ollama", model: "modelo de embeddings" },
        };
        setConfig(nextConfig);
        setSelectedModel(nextConfig.model);

        if (nextConfig.authRequired && !storedToken) {
          setStatus("locked");
          setShowTokenDialog(true);
          return;
        }

        setStatus(await getConnectionStatus(storedToken));
        await loadDocuments(storedToken, currentWorkspaceId);
      } catch {
        if (active) setStatus("offline");
      }
    }

    void bootstrap();
    return () => {
      active = false;
    };
  }, []);

  async function submitMessage(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const message = input.trim();
    if (!message || loading) return;

    const previousMessages = messages;
    const optimisticMessage: UiMessage = { role: "user", content: message };
    setMessages([...previousMessages, optimisticMessage]);
    setInput("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(authToken),
          ...workspaceHeaders(workspaceId),
        },
        body: JSON.stringify({
          message,
          history: previousMessages,
          mode: knowledgeMode,
          documentIds: selectedDocumentIds,
          model: selectedModel || config?.model,
        }),
      });
      const data = await responseData(response);

      if (response.status === 401) {
        window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
        setAuthToken("");
        setTokenDraft("");
        setMessages(previousMessages);
        setInput(message);
        setStatus("locked");
        setShowTokenDialog(true);
        setTokenError("La clave no es válida o ya no está activa.");
        return;
      }

      if (!response.ok) {
        setError(typeof data.error === "string" ? data.error : "No fue posible obtener una respuesta.");
        return;
      }

      if (typeof data.message !== "string") {
        setError("El proveedor respondió sin texto.");
        return;
      }

      setMessages([
        ...previousMessages,
        optimisticMessage,
        {
          role: "assistant",
          content: data.message,
          mode: knowledgeMode,
          cacheHit: data.cacheHit === true,
          embeddingUsed: data.embeddingUsed === true,
          contextTruncated: data.contextTruncated === true,
          sources: parseSources(data.sources),
        },
      ]);
      setStatus("online");
    } catch {
      setStatus("offline");
      setError("No se pudo contactar al gateway. Revisa que la aplicación siga ejecutándose.");
    } finally {
      setLoading(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  async function saveToken(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextToken = tokenDraft.trim();
    if (!nextToken) {
      setTokenError("Escribe la clave de acceso.");
      return;
    }

    window.sessionStorage.setItem(TOKEN_STORAGE_KEY, nextToken);
    setAuthToken(nextToken);
    setTokenError("");
    setStatus("checking");
    const nextStatus = await getConnectionStatus(nextToken);

    if (nextStatus === "locked") {
      window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
      setAuthToken("");
      setTokenError("La clave no es válida.");
      setStatus("locked");
      return;
    }

    setShowTokenDialog(false);
    setStatus(nextStatus);
    await loadDocuments(nextToken, workspaceId);
  }

  async function uploadFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length === 0 || !workspaceId) return;

    setUploading(true);
    setDocumentError("");

    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/documents", {
          method: "POST",
          headers: {
            ...authHeaders(authToken),
            ...workspaceHeaders(workspaceId),
          },
          body: formData,
        });
        const data = await responseData(response);

        if (response.status === 401) {
          setStatus("locked");
          setShowTokenDialog(true);
          setDocumentError("Introduce la clave del gateway para cargar documentos.");
          break;
        }

        if (!response.ok) {
          setDocumentError(typeof data.error === "string" ? data.error : `No fue posible procesar ${file.name}.`);
          continue;
        }

        const nextDocument = data.document;
        if (!nextDocument || typeof nextDocument !== "object" || typeof (nextDocument as Record<string, unknown>).id !== "string") {
          setDocumentError(`El gateway no devolvió información válida para ${file.name}.`);
          continue;
        }

        const summary = nextDocument as DocumentSummary;
        setDocuments((current) => [summary, ...current.filter((document) => document.id !== summary.id)]);
        setSelectedDocumentIds((current) => [...new Set([...current, summary.id])]);
      }
    } catch {
      setDocumentError("No se pudo contactar al gateway para procesar el PDF.");
    } finally {
      setUploading(false);
    }
  }

  function toggleDocument(documentId: string) {
    setSelectedDocumentIds((current) => current.includes(documentId)
      ? current.filter((id) => id !== documentId)
      : [...current, documentId]);
  }

  async function deleteDocument(documentId: string) {
    if (!workspaceId || !window.confirm("¿Quitar este PDF del espacio actual?")) return;

    try {
      const response = await fetch(`/api/documents?id=${encodeURIComponent(documentId)}`, {
        method: "DELETE",
        headers: {
          ...authHeaders(authToken),
          ...workspaceHeaders(workspaceId),
        },
      });
      const data = await responseData(response);
      if (!response.ok) {
        setDocumentError(typeof data.error === "string" ? data.error : "No fue posible quitar el documento.");
        return;
      }

      setDocuments((current) => current.filter((document) => document.id !== documentId));
      setSelectedDocumentIds((current) => current.filter((id) => id !== documentId));
    } catch {
      setDocumentError("No se pudo contactar al gateway para quitar el documento.");
    }
  }

  function clearConversation() {
    if (loading) return;
    setMessages([]);
    setInput("");
    setError("");
    composerRef.current?.focus();
  }

  const providerLabel = config?.provider === "ollama" ? "Ollama" : "vLLM";
  const statusLabel: Record<ConnectionStatus, string> = {
    checking: "Comprobando",
    online: "Conectado",
    offline: "Sin conexión",
    locked: "Protegido",
  };

  return (
    <main className="bridge-shell">
      <section className="bridge-frame" aria-label="LLM Bridge Chat">
        <header className="bridge-header">
          <div className="bridge-brand">
            <div className="bridge-brand-mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div>
              <p className="bridge-eyebrow">LOCAL INFERENCE GATEWAY</p>
              <h1>LLM Bridge</h1>
            </div>
          </div>

          <div className="bridge-header-actions">
            <div className={`connection-pill is-${status}`} aria-live="polite">
              {status === "online" ? <CircleCheck size={14} /> : status === "offline" ? <WifiOff size={14} /> : <span className="connection-dot" />}
              <span>{statusLabel[status]}</span>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="icon-button"
              onClick={clearConversation}
              disabled={loading || messages.length === 0}
              aria-label="Nueva conversación"
              title="Nueva conversación"
            >
              <Plus size={18} />
            </Button>
          </div>
        </header>

        <div className="bridge-meta" aria-label="Configuración activa">
          <span><span className="meta-key">PROVIDER</span> {providerLabel}</span>
          <span className="meta-separator">/</span>
          <span className="model-name" title={config?.model}>{config?.model ?? "cargando configuración"}</span>
          <span className="meta-spacer" />
          {config && config.models.length > 1 && <Select value={selectedModel} onValueChange={setSelectedModel}><SelectTrigger className="knowledge-select" aria-label="Modelo o adaptador"><SelectValue /></SelectTrigger><SelectContent>{config.models.map((model) => <SelectItem value={model} key={model}>{model}</SelectItem>)}</SelectContent></Select>}
          <Link href="/fine-tune" className="token-link"><FlaskConical size={13} /> Fine-tuning</Link>
          <span className="secure-label"><ShieldCheck size={14} /> PDF en memoria · embeddings {config?.embedding.enabled ? "activos" : "desactivados"}</span>
        </div>

        <section className="knowledge-bar" aria-label="Biblioteca documental">
          <div className="knowledge-topline">
            <div className="knowledge-title">
              <span className="meta-key">DOCUMENTOS</span>
              <span className="knowledge-count">{documents.length}/10</span>
            </div>
            <div className="knowledge-actions">
              <Select value={knowledgeMode} onValueChange={(value) => setKnowledgeMode(value as KnowledgeMode)}>
                <SelectTrigger className="knowledge-select" aria-label="Modo de conocimiento">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Sin documentos</SelectItem>
                  <SelectItem value="rag">RAG · buscar fragmentos</SelectItem>
                  <SelectItem value="cag">CAG · contexto completo</SelectItem>
                </SelectContent>
              </Select>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf,.pdf"
                multiple
                onChange={uploadFiles}
                className="sr-only"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="upload-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || !workspaceId || (config?.authRequired === true && !authToken)}
              >
                {uploading ? <LoaderCircle className="spin" size={14} /> : <FileUp size={14} />}
                <span>{uploading ? "Procesando…" : "Agregar PDF"}</span>
              </Button>
            </div>
          </div>

          {documents.length > 0 ? (
            <div className="document-list">
              {documents.map((document) => {
                const selected = selectedDocumentIds.includes(document.id);
                return (
                  <div className={`document-chip${selected ? " is-selected" : ""}`} key={document.id}>
                    <button
                      type="button"
                      className="document-select"
                      onClick={() => toggleDocument(document.id)}
                      aria-pressed={selected}
                      title={`${selected ? "Quitar" : "Usar"} ${document.name}`}
                    >
                      <FileText size={14} />
                      <span className="document-name">{document.name}</span>
                      <span className="document-pages">{document.pages} pág.</span>
                    </button>
                    <button
                      type="button"
                      className="document-delete"
                      onClick={() => void deleteDocument(document.id)}
                      disabled={uploading || loading}
                      aria-label={`Quitar ${document.name}`}
                      title="Quitar PDF"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="knowledge-empty">Agrega un PDF para consultar su contenido desde el chat.</p>
          )}
          <p className="knowledge-hint">
            {knowledgeMode === "cag"
              ? "CAG mantiene en caché el contexto completo; en documentos grandes usa embeddings para elegir una ventana relevante."
              : knowledgeMode === "rag"
                ? "RAG combina embeddings semánticos y coincidencia léxica para recuperar los fragmentos más relacionados."
                : "El chat responderá sin consultar los PDF."}
          </p>
          {documentError && <p className="document-error" role="alert">{documentError}</p>}
        </section>

        <div className="chat-stage" aria-live="polite" aria-busy={loading}>
          {messages.length === 0 ? (
            <div className="empty-chat">
              <div className="empty-icon"><Sparkles size={20} /></div>
              <p className="empty-kicker">LISTO PARA RECIBIR</p>
              <h2>¿Qué quieres explorar?</h2>
              <p className="empty-description">
                {documents.length > 0
                  ? "Selecciona los PDF que quieras consultar y escribe una pregunta."
                  : "Escribe una pregunta y el gateway la enviará al modelo local configurado."}
              </p>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message-row is-${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === "assistant" ? <Bot size={16} /> : <span>TÚ</span>}
                  </div>
                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "assistant"
                        ? `${providerLabel}${message.mode && message.mode !== "none" ? ` · ${message.mode.toUpperCase()}` : ""}`
                        : "TÚ"}
                    </div>
                    <p className="message-content">{message.content}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="message-sources" aria-label="Fuentes consultadas">
                        {[...new Map(message.sources.map((source) => [`${source.documentId}-${source.chunkId}`, source])).values()]
                          .map((source) => (
                            <span className="source-pill" key={`${source.documentId}-${source.chunkId}`}>
                              <FileText size={12} /> {source.documentName} · p. {source.page}{source.pageEnd && source.pageEnd > source.page ? `-${source.pageEnd}` : ""}
                            </span>
                          ))}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {loading && (
                <div className="message-row is-assistant" aria-label="El modelo está respondiendo">
                  <div className="message-avatar" aria-hidden="true"><Bot size={16} /></div>
                  <div className="message-body">
                    <div className="message-label">{providerLabel}</div>
                    <div className="thinking-indicator"><span /><span /><span /></div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <CircleAlert size={16} />
            <span>{error}</span>
          </div>
        )}

        <form className="composer" ref={formRef} onSubmit={submitMessage}>
          <Textarea
            ref={composerRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Escribe un mensaje…"
            aria-label="Mensaje"
            rows={1}
            disabled={loading}
            className="composer-input"
          />
          <div className="composer-footer">
            <span className="composer-hint">Enter para enviar · Shift + Enter para saltar línea</span>
            <Button type="submit" size="lg" disabled={!input.trim() || loading || (config?.authRequired === true && !authToken)}>
              {loading ? <LoaderCircle className="spin" size={16} /> : <ArrowUp size={16} />}
              <span>Enviar</span>
            </Button>
          </div>
        </form>

        <footer className="bridge-footer">
          <span><span className="footer-mark" /> El proveedor se ejecuta en tu computador</span>
          {config?.authRequired && (
            <button type="button" className="token-link" onClick={() => setShowTokenDialog(true)}>
              <KeyRound size={13} /> Cambiar clave
            </button>
          )}
        </footer>
      </section>

      <Dialog open={showTokenDialog} onOpenChange={setShowTokenDialog}>
        <DialogContent className="bridge-dialog">
          <DialogHeader>
            <DialogTitle>Clave del gateway</DialogTitle>
            <DialogDescription>
              La aplicación está protegida porque puede quedar expuesta mediante un túnel público. La clave solo se guarda en esta sesión del navegador.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={saveToken} className="token-form">
            <label htmlFor="gateway-token">Clave de acceso</label>
            <input
              id="gateway-token"
              type="password"
              autoComplete="current-password"
              value={tokenDraft}
              onChange={(event) => setTokenDraft(event.target.value)}
              className="token-input"
              placeholder="APP_TOKEN"
              autoFocus
            />
            {tokenError && <p className="token-error" role="alert">{tokenError}</p>}
            <DialogFooter>
              <Button type="submit">Continuar</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </main>
  );
}
