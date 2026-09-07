import { addDocument, getDocumentConfig, indexPdf, listDocuments, removeDocument } from "@/lib/documents";
import { errorResponse, isAuthorized, workspaceIdFrom } from "@/lib/http/request";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!isAuthorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  const workspaceId = workspaceIdFrom(request);
  if (!workspaceId) return errorResponse("El espacio de documentos no es válido.", 400);

  return Response.json({ documents: listDocuments(workspaceId) });
}

export async function POST(request: Request) {
  if (!isAuthorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  const workspaceId = workspaceIdFrom(request);
  if (!workspaceId) return errorResponse("El espacio de documentos no es válido.", 400);

  const documentConfig = getDocumentConfig();
  if (listDocuments(workspaceId).length >= documentConfig.maxDocuments) {
    return errorResponse(
      `Este espacio ya contiene el máximo de ${documentConfig.maxDocuments} documentos.`,
      409,
    );
  }

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
  if (value.size > documentConfig.maxPdfBytes) {
    return errorResponse(`El PDF supera el límite de ${Math.round(documentConfig.maxPdfBytes / 1024 / 1024)} MB.`, 413);
  }

  try {
    const document = await indexPdf(new Uint8Array(await value.arrayBuffer()), value.name, value.size);
    const summary = addDocument(workspaceId, document);
    return Response.json({ document: summary }, { status: 201 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "No fue posible procesar el PDF.";
    console.error("[llm-bridge] PDF indexing failed", error);
    const status = /máximo de .*documentos/i.test(message)
      ? 409
      : /supera el límite/i.test(message)
        ? 413
        : 422;
    return errorResponse(message, status);
  }
}

export async function DELETE(request: Request) {
  if (!isAuthorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

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
