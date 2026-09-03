"use client";

import Link from "next/link";
import { useMemo, useState, type FormEvent } from "react";
import { ArrowLeft, CheckCircle2, FlaskConical, LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

type Validation = { valid: boolean; examples: number; messages: number; characters: number; errors: string[] };
const TOKEN_STORAGE_KEY = "llm-bridge.app-token";

export default function FineTunePage() {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("qwen-es-v1");
  const [rank, setRank] = useState("16");
  const [epochs, setEpochs] = useState("3");
  const [validation, setValidation] = useState<Validation | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const command = useMemo(() => file && validation?.valid
    ? `./scripts/train-adapter.sh ${JSON.stringify(file.name)} ${JSON.stringify(name)} --rank ${rank} --epochs ${epochs}`
    : "", [file, name, rank, epochs, validation]);

  async function validate(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setLoading(true); setError(""); setValidation(null);
    const form = new FormData(); form.append("file", file);
    const token = window.sessionStorage.getItem(TOKEN_STORAGE_KEY) ?? "";
    try {
      const response = await fetch("/api/fine-tune/validate", { method: "POST", headers: token ? { Authorization: `Bearer ${token}` } : {}, body: form });
      const data = await response.json() as Validation & { error?: string };
      if (!response.ok) setError(data.error ?? data.errors?.[0] ?? "Dataset inválido.");
      setValidation(data.valid === true ? data : null);
    } catch { setError("No fue posible validar el dataset."); }
    finally { setLoading(false); }
  }

  return (
    <main className="min-h-screen bg-[#090b0c] px-5 py-10 text-zinc-100">
      <section className="mx-auto max-w-3xl rounded-2xl border border-zinc-800 bg-[#101314] p-6 shadow-2xl">
        <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-100"><ArrowLeft size={15} /> Volver al chat</Link>
        <div className="mb-7 flex items-start gap-4">
          <div className="rounded-xl border border-emerald-900 bg-emerald-950/40 p-3 text-emerald-400"><FlaskConical size={22} /></div>
          <div><p className="text-xs tracking-[.18em] text-emerald-500">QLORA · SFT</p><h1 className="mt-1 text-2xl font-semibold">Preparar fine-tuning</h1><p className="mt-2 text-sm leading-6 text-zinc-400">Valida el dataset y genera el comando reproducible. El entrenamiento se ejecuta localmente con vLLM detenido para reservar la GPU.</p></div>
        </div>
        <form onSubmit={validate} className="grid gap-5">
          <label className="grid gap-2 text-sm">Dataset conversacional JSONL<Input type="file" accept=".jsonl,application/json" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setValidation(null); }} required /></label>
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="grid gap-2 text-sm">Nombre del adaptador<Input value={name} pattern="[A-Za-z0-9][A-Za-z0-9._-]{0,63}" onChange={(event) => setName(event.target.value)} required /></label>
            <label className="grid gap-2 text-sm">Rank<Select value={rank} onValueChange={setRank}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="8">8</SelectItem><SelectItem value="16">16</SelectItem><SelectItem value="32">32</SelectItem></SelectContent></Select></label>
            <label className="grid gap-2 text-sm">Épocas<Input type="number" min="0.1" max="20" step="0.1" value={epochs} onChange={(event) => setEpochs(event.target.value)} required /></label>
          </div>
          <Button type="submit" disabled={!file || loading}>{loading ? <LoaderCircle className="animate-spin" size={16} /> : <CheckCircle2 size={16} />} Validar dataset</Button>
        </form>
        {error && <p className="mt-5 rounded-lg border border-red-900 bg-red-950/30 p-3 text-sm text-red-300">{error}</p>}
        {validation?.valid && <div className="mt-6 rounded-xl border border-emerald-900/70 bg-emerald-950/20 p-5"><p className="font-medium text-emerald-300">Dataset válido: {validation.examples} ejemplos · {validation.messages} mensajes · {validation.characters.toLocaleString("es-CL")} caracteres</p><p className="mt-4 text-sm text-zinc-400">Copia el archivo a la raíz del proyecto y ejecuta:</p><pre className="mt-2 overflow-x-auto rounded-lg bg-black p-4 text-sm text-emerald-300"><code>{command}</code></pre><p className="mt-3 text-xs leading-5 text-zinc-500">Al terminar, agrega <code>{name}</code> a <code>LLM_ADAPTER_MODELS</code> y precárgalo en vLLM con <code>--lora-modules {name}=/adapters/{name}</code>.</p></div>}
      </section>
    </main>
  );
}
