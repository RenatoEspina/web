import { validateJsonlDataset } from "@/lib/fine-tuning/dataset";
import { getAppToken } from "@/lib/llm/config";

export const dynamic = "force-dynamic";
const MAX_DATASET_BYTES = 25 * 1024 * 1024;

function authorized(request: Request): boolean {
  const expected = getAppToken();
  if (!expected) return true;
  const authorization = request.headers.get("authorization") ?? "";
  const supplied = authorization.startsWith("Bearer ") ? authorization.slice(7).trim() : request.headers.get("x-app-token") ?? "";
  return supplied === expected;
}

export async function POST(request: Request) {
  if (!authorized(request)) return Response.json({ error: "Se requiere una clave de acceso." }, { status: 401 });
  const form = await request.formData();
  const file = form.get("file");
  if (!(file instanceof File)) return Response.json({ error: "Adjunta un archivo JSONL en el campo file." }, { status: 400 });
  if (file.size > MAX_DATASET_BYTES) return Response.json({ error: "El dataset supera el límite de 25 MB." }, { status: 413 });
  const validation = validateJsonlDataset(await file.text());
  return Response.json(validation, { status: validation.valid ? 200 : 422 });
}
