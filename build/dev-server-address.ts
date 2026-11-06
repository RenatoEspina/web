import { loadEnv } from "vite";

export type DevServerAddress = {
  host: string;
  port: number;
};

/**
 * Resolves the dev server bind address from the process environment and the
 * `.env*` files of `envDir`. Vite does not load those files into
 * `process.env` while evaluating its own config, so `HOST`/`PORT` written only
 * in `.env.local` would otherwise be ignored. Process variables keep priority
 * over file values, matching `loadEnv` semantics.
 */
export function resolveDevServerAddress(mode: string, envDir: string): DevServerAddress {
  const env = loadEnv(mode, envDir, "");
  return {
    host: env.HOST ?? "127.0.0.1",
    port: Number(env.PORT ?? 3000),
  };
}
