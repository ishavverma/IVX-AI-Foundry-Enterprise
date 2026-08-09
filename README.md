# AI Foundry Enterprise (IVX)

**AI Foundry Enterprise** (the *IVX* repository) is a **modular AI operating platform** that provides a full‑stack foundation for building, training, fine‑tuning, evaluating, and serving large language models (LLMs) as well as the surrounding ecosystem (datasets, tokenizers, plugins, agents, MCP, monitoring, etc.).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Layers](#architecture-layers)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Development Server](#running-the-development-server)
4. [Key Components](#key-components)
5. [Model Factory](#model-factory)
6. [Dataset Pre‑training Resources](#dataset-pre‑training-resources)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview

The repository follows the **AI Foundry OS** roadmap (see `roadmap.txt`).  It is organized as a **single‑repo monolith** that can later be split into independent services.  The main goals are:

* Provide a **stable core platform** (authentication, RBAC, storage, resource management, event bus, queue).
* Offer **engine modules** for tokenization, model definition, training, fine‑tuning, evaluation, inference, and deployment.
* Deliver **experience‑layer apps** (dashboards, chat, agents, MCP, monitoring, collaboration, marketplace).
* Keep everything **extensible** via a plugin SDK so the community can add new architectures, optimizers, data sources, etc.

---

## Architecture Layers

```
ai_foundry_enterprise/
│
├─ apps/                # Flask blueprints (UI & API) – Model Factory, Datasets, Training, …
├─ core/                # Platform services – config, DB, storage, security, queue, events
├─ engines/             # AI engine implementations – tokenizer, model, training, inference
├─ workers/             # Background workers for async jobs (dataset pipelines, training)
├─ plugins/             # Plugin SDK and loaded plugins
├─ api/                 # Public REST API (future external services)
├─ ui/                  # Jinja templates, static assets (CSS/JS)
├─ docs/                # Design docs, architecture diagrams
└─ tests/               # Pytest suite
```

Each layer communicates through well‑defined interfaces (SQLAlchemy models, storage SDK, event bus) which makes it straightforward to extract services into separate containers later.

---

## Getting Started

### Prerequisites

* **Python 3.11+**
* **Poetry** (or pip‑env) for dependency management – the project uses a `pyproject.toml`.
* **Docker & Docker‑Compose** (optional, for running the full stack with PostgreSQL, Redis, etc.)
* **Git**

### Installation

```bash
# Clone the repository
git clone https://github.com/your‑org/ivx.git
cd ivx

# Install dependencies via Poetry (recommended)
poetry install

# Or with pip if you prefer
pip install -r requirements.txt
```

### Running the Development Server

The platform uses **Flask** for the web UI and **SQLAlchemy** with SQLite for quick local development.

```bash
# Initialise the database (creates tables)
flask db upgrade   # if using Flask‑Migrate, otherwise run the provided script

# Start the Flask development server
flask run --reload
```

Open your browser at `http://localhost:5000` – you will see the dashboard with navigation links to **Model Factory**, **Datasets**, **Training**, etc.

## UI Redesign – Phase 1 (Foundations)

The **Phase 1 UI redesign** focuses on security, accessibility, and a component‑first architecture.

### Highlights

* **Security hardening** – Content‑Security‑Policy, HSTS, Referrer‑Policy, Permissions‑Policy, and `SESSION_COOKIE_SECURE` are now set in `app.py`.
* **Global CSS reset** – `normalize.css` is loaded via CDN.
* **Utility‑first styling** – Tailwind CSS is included via CDN for rapid UI prototyping.
* **Component library** – Reusable Jinja macros live in `ui/templates/_components.html` (buttons, cards, tables, forms, modals, toasts, pagination, badges, avatars, progress bars, and a `form` macro that automatically injects the CSRF token).
* **Template refactor** – Core templates (`auth/login.html`, `auth/register.html`, `agents/new.html`, `admin/users.html`, `datasets/new.html`, etc.) now use the component macros for consistent markup, ARIA labels, and client‑side validation.
* **Dark‑mode toggle** – Persisted via `localStorage` and toggled with the theme button in the navbar.

### Building the UI

Phase 2 uses a local Tailwind build so production pages do not depend on a CDN:

```bash
npm install
npm run build:css
```

Run `npm run watch:css` during template work. The generated `ui/static/css/tailwind.css` is intentionally a build artifact and should be produced as part of deployment or CI.

### Testing

Added a basic Flask test (`tests/test_ui_security.py`) that verifies the presence of the CSRF hidden input on forms and checks that security headers are present in responses.

Run the UI tests with:

```bash
pytest tests/test_ui_security.py -q
```

---

## Key Components

| Component | Package | Description |
|-----------|---------|-------------|
| **Core Platform** | `core/` | Config, DB, storage SDK, RBAC, event bus, queue, security |
| **Model Factory** | `apps/model_factory/` | Visual architecture composer, parameter/VRAM calculator, PyTorch exporter, registry integration |
| **Dataset Studio** | `apps/datasets/` | Ingestion, validation, cleaning, chunking, quality scoring |
| **Tokenizer Engine** | `engines/tokenizer/` | SentencePiece/BPE/Unigram implementations |
| **Training Engine** | `engines/training/` | Trainer API (PyTorch, Accelerate, DeepSpeed) |
| **Fine‑tuning Engine** | `engines/finetuning/` | LoRA, QLoRA, adapters, RLHF pipelines |
| **Evaluation Engine** | `engines/evaluation/` | Benchmarks (MMLU, HumanEval, etc.) |
| **Inference Engine** | `engines/inference/` | vLLM, SGLang, TensorRT, OpenAI‑compatible API |
| **Agent Studio** | `apps/agents/` | Planner, memory, function calling, RAG, workflow orchestration |
| **MCP Studio** | `apps/mcp/` | Model‑Component‑Package registry, tool publishing |

---

## Model Factory

The **Model Factory** lets you design neural architectures via a wizard UI (`/models`).  It:

* Offers preset tiers (Nano → MoE‑1T) with pre‑computed FLOPs/VRAM.
* Calculates parameters, active parameters, VRAM (FP16/FP8/INT4) and GFLOPs on‑the‑fly.
* Generates a **PyTorch model file** and a JSON configuration that are stored in the central storage SDK.
* Emits a `model_created` event so downstream services (training, registry) can react automatically.

Visit `/models` to explore the new dashboard and start building your own model definitions.

---

## Dataset Pre‑training Resources

A curated list of ~4‑6 GB datasets suitable for initial LLM pre‑training is available at **`/datasets/pretrain`**.  The page lists:

* **OpenWebText2** – ~5 GB of Reddit‑derived web text.
* **The Pile – Small Slice** – ~4.5 GB mixed‑domain data.
* **C4 Filtered** – ~5 GB cleaned Common Crawl data.

Each entry includes a short description, source URL, and a direct download button via the 🤗 Datasets hub.

---

## Testing

The repository ships with a comprehensive pytest suite covering core services, engine calculations, and UI routes.

```bash
pytest -q
```

Run the tests after any change to ensure nothing is broken.

---

## Deployment

For production you can use the provided **Docker Compose** configuration (`docker-compose.yml`).  It brings up:

* PostgreSQL (or MySQL) for persistent storage
* Redis for the task queue
* The Flask app behind **Gunicorn**
* Optional **Prometheus** & **Grafana** for monitoring (see `deploy/` folder)

```bash
docker compose up -d --build
```

---

## Contributing

Contributions are welcome!  Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your‑feature`).
3. Write tests for your changes.
4. Ensure code style passes `ruff`/`black` (the repo includes a pre‑commit config).
5. Open a Pull Request with a clear description of the change.

See `CONTRIBUTING.md` for detailed guidelines (to be added soon).

---

## License

This project is licensed under the **Apache License 2.0** – see the `LICENSE` file for details.

---

*Happy building!*