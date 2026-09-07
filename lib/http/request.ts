import { getAppToken } from "@/lib/llm/config";

const WORKSPACE_ID_PATTERN = /^[A-Za-z0-9_-]{16,80}$/;

function suppliedAppToken(request: Request): string {
  const authorization = request.headers.get("authorization") ?? "";
  if (authorization.startsWith("Bearer ")) {
    return authorization.slice("Bearer ".length).trim();
  }
  return request.headers.get("x-app-token")?.trim() ?? "";
}

export function isAuthorized(request: Request): boolean {
  const appToken = getAppToken();
  return !appToken || suppliedAppToken(request) === appToken;
}

export function workspaceIdFrom(request: Request): string | null {
  const workspaceId = request.headers.get("x-workspace-id")?.trim() ?? "";
  return WORKSPACE_ID_PATTERN.test(workspaceId) ? workspaceId : null;
}

export function errorResponse(message: string, status: number): Response {
  return Response.json({ error: message }, { status });
}
