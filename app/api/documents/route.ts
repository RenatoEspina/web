import { addDocument, indexPdf, listDocuments, removeDocument } from "@/lib/documents";
import { getAppToken } from "@/lib/llm/config";

export const dynamic = "force-dynamic";

const WORKSPACE_ID_PATTERN = /^[A-Za-z0-9_-]{16,80}$/;

function authorized(request: Request): boolean {
  const appToken = getAppToken();
  if (!appToken) return true;

  const authorization = request.headers.get("authorization") ?? "";
  const suppliedToken = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : request.headers.get("x-app-token") ?? "";

  return suppliedToken === appToken;
}

function workspaceIdFrom(request: Request): string | null {
  const workspaceId = request.headers.get("x-workspace-id")?.trim() ?? "";
  return WORKSPACE_ID_PATTERN.test(workspaceId) ? workspaceId : null;
}

function errorResponse(message: string, status: number) {
  return Response.json({ error: message }, { status });
}

export async function GET(request: Request) {
  if (!authorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  const workspaceId = workspaceIdFrom(request);
  if (!workspaceId) return errorResponse("El espacio de documentos no es válido.", 400);

  return Response.json({ documents: listDocuments(workspaceId) });
}

export async function POST(request: Request) {
  if (!authorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  const workspaceId = workspaceIdFrom(request);
  if (!workspaceId) return errorResponse("El espacio de documentos no es válido.", 400);

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return errorResponse("La carga debe utilizar un formulario multipart/form-data.", 400);
  }

  const value = form.get("file");
  if (!(value instanceof File)) return errorResponse("Selecciona un archivo PDF.", 400);

  const isPdf = value.type === "application/pdf" || value.name.toLocaleLowerCase().endsWith(".pdf");
  if (!isPdf) return errorResponse("Solo se admiten archivos PDF.", 415);
  if (value.size === 0) return errorResponse("El archivo PDF está vacío.", 400);

  try {
    const document = await indexPdf(new Uint8Array(await value.arrayBuffer()), value.name, value.size);
    const summary = addDocument(workspaceId, document);
    return Response.json({ document: summary }, { status: 201 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "No fue posible procesar el PDF.";
    console.error("[llm-bridge] PDF indexing failed", error);
    const status = /supera el límite|máximo de/i.test(message) ? 413 : 422;
    return errorResponse(message, status);
  }
}

export async function DELETE(request: Request) {
  if (!authorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  const workspaceId = workspaceIdFrom(request);
  if (!workspaceId) return errorResponse("El espacio de documentos no es válido.", 400);

  const documentId = new URL(request.url).searchParams.get("id")?.trim() ?? "";
  if (!documentId || !/^[A-Za-z0-9-]{16,100}$/.test(documentId)) {
    return errorResponse("El documento indicado no es válido.", 400);
  }

  if (!removeDocument(workspaceId, documentId)) {
    return errorResponse("El documento no existe en este espacio.", 404);
  }

  return Response.json({ ok: true });
}
