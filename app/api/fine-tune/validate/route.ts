import { validateJsonlDataset } from "@/lib/fine-tuning/dataset";
import { errorResponse, isAuthorized } from "@/lib/http/request";

export const dynamic = "force-dynamic";
const MAX_DATASET_BYTES = 25 * 1024 * 1024;

export async function POST(request: Request) {
  if (!isAuthorized(request)) return errorResponse("Se requiere una clave de acceso.", 401);

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return errorResponse("La carga debe utilizar un formulario multipart/form-data.", 400);
  }

  const file = form.get("file");
  if (!(file instanceof File)) return errorResponse("Adjunta un archivo JSONL en el campo file.", 400);
  if (file.size > MAX_DATASET_BYTES) return errorResponse("El dataset supera el límite de 25 MB.", 413);

  const validation = validateJsonlDataset(await file.text());
  return Response.json(validation, { status: validation.valid ? 200 : 422 });
}
