# DeepFake Forensics AI

DeepFake Forensics AI is a presentation-ready academic prototype for multimodal media forensics. Upload an image, audio clip, or video to create a traceable case file with a deterministic prototype prediction, confidence, risk level, signal charts, evidence notes, explainability, and downloadable reports.

> This build is explicitly **Prototype inference mode**. It does not claim benchmark accuracy and does not replace a validated forensic model.

## Features

- Image, audio, and video upload flows
- Real multipart file handling with safe filenames and per-case local folders
- Deterministic file-derived feature extraction
- REAL / FAKE prototype prediction, confidence, and LOW / MEDIUM / HIGH risk
- Signal profile charts and evidence timeline
- Searchable analysis history with delete actions
- JSON, CSV, lightweight PDF, original-file, and evidence downloads
- Responsive workbench, result, history, report, and methodology screens
- Honest limitations shown throughout the UI

## Architecture

- Frontend: React, TypeScript, Vite, Tailwind CSS, Wouter, Recharts, Lucide React
- API: Express 5 in the shared API server artifact
- Contract: `lib/api-spec/openapi.yaml`, with generated clients in `lib/api-client-react`
- Runtime storage: `storage/jobs/<analysis-id>/original` and `storage/jobs/<analysis-id>/evidence`
- History: in-memory for the running academic prototype session

The original brief called for FastAPI/Python and heavyweight ML packages. The workspace runtime is a pnpm/TypeScript monorepo, so this first working build uses the shared Express service and a lightweight deterministic pipeline. No trained model is presented as active. The API surface is intentionally structured so a validated Python inference service can replace the prototype scoring layer later.

## Run in this workspace

Start the API:

```bash
pnpm --filter @workspace/api-server run dev
```

Start the frontend:

```bash
pnpm --filter @workspace/deepfake-forensics-ai run dev
```

Run the workspace typecheck:

```bash
pnpm run typecheck
```

Build the frontend manually outside the managed workflow:

```bash
PORT=5173 BASE_PATH=/ pnpm --filter @workspace/deepfake-forensics-ai run build
```

The managed workflows provide `PORT` and `BASE_PATH` automatically.

## Windows setup

Install Node.js 20 or newer and pnpm:

```powershell
corepack enable
corepack prepare pnpm@latest --activate
pnpm install
```

Run the API:

```powershell
pnpm --filter @workspace/api-server run dev
```

Run the frontend in a second PowerShell window:

```powershell
pnpm --filter @workspace/deepfake-forensics-ai run dev
```

No FFmpeg installation is required for the current lightweight prototype because it does not decode media streams. A future validated video/audio extractor can add FFmpeg and Python dependencies behind the same API contract.

## Environment variables

Copy `.env.example` when running outside the managed workspace:

```bash
cp .env.example .env
```

- `DEMO_MODE=true` — keeps the prototype inference label enabled.
- `VITE_API_URL=/api` — optional frontend API prefix; the managed preview routes `/api` automatically.
- `PORT` and `BASE_PATH` — supplied by the managed workflows.

## API routes

- `GET /api/healthz`
- `GET /api/system-info`
- `GET /api/analyses`
- `GET /api/analyses/:analysisId`
- `DELETE /api/analyses/:analysisId`
- `POST /api/analyze/image`
- `POST /api/analyze/audio`
- `POST /api/analyze/video`
- `POST /api/reports/:analysisId/generate`
- `GET /api/reports/:analysisId/json`
- `GET /api/reports/:analysisId/csv`
- `GET /api/reports/:analysisId/pdf`
- `GET /api/reports/:analysisId/package`
- `GET /api/evidence/:analysisId/original`
- `GET /api/evidence/:analysisId/evidence`

## Reports and downloads

The result page links to JSON, CSV, and PDF reports plus the original upload and generated evidence JSON. The report desk also returns a complete ZIP package containing `report.json`, `report.pdf`, `metadata.json`, and `evidence/analysis.json`. PDF output is intentionally lightweight and self-contained so it works without a document-generation dependency. Report links are scoped by UUID analysis IDs; arbitrary filesystem paths are never accepted.

## Prototype limitations

- Scores are deterministic byte-level proxies, not trained DeepFake classifiers.
- No face detector, ViT, optical flow, MFCC decoder, or validated lip-sync model is active.
- Video temporal and audio-visual values are clearly labeled proxy signals.
- History and uploaded files are local to the current API process and workspace.
- The prototype has a 25 MB upload limit.
- Corrupted or unsupported media receives a user-facing validation error; it is not a benchmarked forensic conclusion.

## Future improvements

1. Add a Python inference service with validated model adapters.
2. Add OpenCV/FFmpeg frame and audio extraction behind feature flags.
3. Add a real face detector and validated temporal/audio-visual metrics.
4. Move persistent media to object storage and history to PostgreSQL.
5. Add background jobs, cleanup policies, automated tests, and evidence ZIP generation.

## Exporting the project source

Use the workspace's project download/export action to download the complete source tree. The important implementation directories are:

```text
artifacts/deepfake-forensics-ai/
artifacts/api-server/
lib/api-spec/
lib/api-client-react/
storage/jobs/
```