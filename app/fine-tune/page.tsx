import { ArrowLeft, FlaskConical, Terminal } from "lucide-react";
import Link from "next/link";

export default function FineTunePage() {
  return (
    <main className="min-h-screen bg-[#090b0c] px-5 py-10 text-zinc-100">
      <section className="mx-auto max-w-3xl rounded-2xl border border-zinc-800 bg-[#101314] p-6 shadow-2xl">
        <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-100">
          <ArrowLeft size={15} /> Volver al chat
        </Link>

        <div className="mb-7 flex items-start gap-4">
          <div className="rounded-xl border border-emerald-900 bg-emerald-950/40 p-3 text-emerald-400">
            <FlaskConical size={22} />
          </div>
          <div>
            <p className="text-xs tracking-[.18em] text-emerald-500">QLORA · SFT</p>
            <h1 className="mt-1 text-2xl font-semibold">Fine-tuning local</h1>
            <p className="mt-2 text-sm leading-6 text-zinc-400">
              El flujo de fine-tuning se administra desde una GUI separada que solo escucha en loopback.
              Así el chat público no expone operaciones administrativas, acceso a Docker ni control de la GPU.
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-black/30 p-5">
          <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
            <Terminal size={16} /> Iniciar la GUI local
          </div>
          <pre className="mt-3 overflow-x-auto rounded-lg bg-black p-4 text-sm text-emerald-300">
            <code>./fine-tune-gui</code>
          </pre>
          <p className="mt-3 text-sm leading-6 text-zinc-400">
            Desde esa interfaz puedes subir y validar datasets, preparar el entorno, entrenar adaptadores,
            iniciar o detener vLLM y cargar o descargar LoRA dinámicamente.
          </p>
        </div>

        <p className="mt-5 text-xs leading-5 text-zinc-500">
          También puedes usar <code>./comandos.fish fine-tune-help</code> para operar el flujo desde terminal.
        </p>
      </section>
    </main>
  );
}
