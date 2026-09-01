"use client";

import {
  ArrowUp,
  Bot,
  CircleAlert,
  CircleCheck,
  KeyRound,
  LoaderCircle,
  Plus,
  ShieldCheck,
  Sparkles,
  WifiOff,
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";

type UiMessage = {
  role: "user" | "assistant";
  content: string;
};

type GatewayConfig = {
  provider: "vllm" | "ollama";
  model: string;
  authRequired: boolean;
};

type ConnectionStatus = "checking" | "online" | "offline" | "locked";

const TOKEN_STORAGE_KEY = "llm-bridge.app-token";

function authHeaders(token: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
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
  const [authToken, setAuthToken] = useState("");
  const [tokenDraft, setTokenDraft] = useState("");
  const [showTokenDialog, setShowTokenDialog] = useState(false);
  const [tokenError, setTokenError] = useState("");
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    let active = true;
    const storedToken = window.sessionStorage.getItem(TOKEN_STORAGE_KEY) ?? "";

    async function bootstrap() {
      try {
        const response = await fetch("/api/config", { cache: "no-store" });
        const data = await responseData(response);
        if (!active) return;
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
        };
        setConfig(nextConfig);

        if (nextConfig.authRequired && !storedToken) {
          setStatus("locked");
          setShowTokenDialog(true);
          return;
        }

        setStatus(await getConnectionStatus(storedToken));
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
        },
        body: JSON.stringify({ message, history: previousMessages }),
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

      setMessages([...previousMessages, optimisticMessage, { role: "assistant", content: data.message }]);
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
          <span className="secure-label"><ShieldCheck size={14} /> Solo texto · sin persistencia</span>
        </div>

        <div className="chat-stage" aria-live="polite" aria-busy={loading}>
          {messages.length === 0 ? (
            <div className="empty-chat">
              <div className="empty-icon"><Sparkles size={20} /></div>
              <p className="empty-kicker">LISTO PARA RECIBIR</p>
              <h2>¿Qué quieres explorar?</h2>
              <p className="empty-description">Escribe una pregunta y el gateway la enviará al modelo local configurado.</p>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message-row is-${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === "assistant" ? <Bot size={16} /> : <span>TÚ</span>}
                  </div>
                  <div className="message-body">
                    <div className="message-label">{message.role === "assistant" ? providerLabel : "TÚ"}</div>
                    <p className="message-content">{message.content}</p>
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
