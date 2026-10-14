# DeepFake Forensics AI

An academic multimodal media-forensics prototype that turns image, audio, and video uploads into traceable, explainable case files.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/deepfake-forensics-ai/src/` — responsive React/Vite application, routes, upload flows, charts, evidence, and report actions.
- `artifacts/api-server/src/routes/analyses.ts` — upload parsing, deterministic prototype scoring, local job storage, history, and evidence downloads.
- `artifacts/api-server/src/routes/reports.ts` and `store.ts` — JSON, CSV, and lightweight PDF report downloads.
- `lib/api-spec/openapi.yaml` — source of truth for generated API client and validation schemas.
- `storage/jobs/` — runtime-local per-analysis original and evidence files.

## Architecture decisions

- The first build uses a deterministic byte-feature prototype instead of claiming trained DeepFake model inference. Every result and report is labeled Prototype inference mode.
- Analysis records are kept in memory for the running prototype session; uploaded files and generated evidence are written to job-specific local folders, not stored in PostgreSQL.
- The API returns normalized 0–1 confidence and score values because the frontend renders them as percentages.
- The server accepts multipart uploads without a native media dependency, keeping the first build runnable while preserving real file handling and safe filename/path boundaries.

## Product

Users can upload image, audio, or video files; review prototype REAL/FAKE predictions, risk, normalized signals, charts, metadata, evidence markers, and explanations; search/delete case history; and download JSON, CSV, PDF, original, or evidence files.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Frontend builds need the workflow-provided `PORT` and `BASE_PATH`; for a manual build use `PORT=5173 BASE_PATH=/ pnpm --filter @workspace/deepfake-forensics-ai run build`.
- Restart both managed workflows after API or frontend changes so the proxy and generated client stay aligned.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
