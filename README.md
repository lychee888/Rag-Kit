# rag-kit

Agentic RAG system for Hermes Agent. Watches a folder, auto-ingests documents (PDF, DOCX, XLSX, PPTX, TXT, MD, and images via OCR), embeds with a multilingual sentence-transformers model, stores in LanceDB, and answers via a CLI designed for LLM agents.

## Quickstart (one-liner per platform)

```bash
# Windows
scripts\install-windows.bat

# Linux / DGX Spark
chmod +x scripts/install-dgx-spark.sh && ./scripts/install-dgx-spark.sh
```

That's it. The installer creates a venv at `~/rag-kit-venv/`, installs everything, starts the watcher, and optionally sets up autostart.

## Offline Installation

1. Download models first (on a machine with internet):
   ```bash
   # Windows
   scripts\download-models.bat

   # Linux
   chmod +x scripts/download-models.sh && ./scripts/download-models.sh
   ```

2. Copy `~/models/` to the target machine, then run the install script — it works fully offline.

## Features

- **Auto-ingest**: Watches a folder, processes new/changed/deleted files automatically
- **Multilingual**: Chinese + English text extraction, chunking, and embeddings
- **OCR fallback**: EasyOCR for scanned PDFs and image-based documents
- **VLM captioning**: SmolVLM-256M for charts and diagrams (optional, configurable)
- **Hybrid search**: Semantic (vector similarity) + keyword (FTS) scoring via LanceDB
- **Local vector DB**: LanceDB — no server, no external dependencies
- **Agent-friendly**: All CLI commands support `--json` output for Hermes Agent
- **2 GB budget**: Total model memory (embedding + OCR + VLM) fits within 2 GB
- **Cross-platform**: Windows + Linux (DGX Spark ARM64)
- **Autostart**: Windows Scheduled Task or Linux systemd user service on boot

## CLI Reference

| Command | Description | JSON flag |
| `rag ingest <path>` | Ingest a file or folder | `--json` |
| `rag query "<text>"` | Hybrid search (vector + keyword) | `--json` |
| `rag list-files` | List all ingested files | `--json` |
| `rag delete <path>` | Remove a file's chunks from DB | `--json` |
| `rag status` | Show system status and model info | `--json` |
| `rag watch <folder>` | Start real-time folder watcher | `--json` |
| `rag setup-autostart` | Configure autostart (boot) | `--json` |
| `rag config` | Print or update configuration | — |
| `rag config init` | Create default config file | — |
| `rag config path` | Show config file location | — |
| `rag config set <key> <value>` | Update a config value | — |
| `rag --version` | Show version | — |

### JSON output examples

```bash
# Search (agent consumption)
rag query --json "机器学习是什么" --top-k 5

# System status
rag status --json

# List files
rag list-files --json

# Setup autostart
rag setup-autostart --json
```

## Configuration

rag-kit loads config from (in order of precedence):

1. `RAG_KIT_CONFIG` environment variable (path to a YAML file)
2. `./rag-kit.yaml` (current directory)
3. `~/.rag-kit.yaml` (user home)

Every config key can be overridden via environment variable `RAG_KIT_<KEY>`:

```bash
export RAG_KIT_WATCH_FOLDER=~/my-documents
export RAG_KIT_DB_PATH=~/my-lancedb
export RAG_KIT_VLM_ENABLED=false
export RAG_KIT_HF_ENDPOINT=https://hf-mirror.com   # China mirror
```

### Config file reference

| Key | Type | Default | Description |
| `watch_folder` | path | `~/Documents/rag-ingest` | Folder the watcher monitors |
| `db_path` | path | `~/lancedb` | LanceDB storage directory |
| `model_dir` | path | `~/models` | Model cache directory |
| `embedding_model` | str | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model |
| `vlm_model` | str | `HuggingFaceTB/SmolVLM-256M-Instruct` | VLM model for captions |
| `vlm_enabled` | bool | `true` | Enable VLM captioning |
| `supported_extensions` | list | `[.pdf, .docx, .xlsx, .pptx, .txt, .md, .png, .jpg, .jpeg, .tiff, .webp]` | File types to ingest |
| `languages` | list | `[zh, en]` | OCR languages |
| `chunk_size` | int | `512` | Target chunk length (chars) |
| `chunk_overlap` | int | `64` | Overlap between chunks |
| `watch_interval` | int | `30` | Reserved. Watcher is state-change based (watchdog) — needs no poll interval |
| `search_alpha` | float | `0.5` | Semantic vs keyword blend |
| `max_memory_mb` | int | `2048` | Memory budget guard |
| `hf_endpoint` | str | `""` | HF endpoint (set for China) |


```

### Models

| Component | Model | Disk | Runtime RAM |
|---|---|---|---|
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | ~470 MB | ~500 MB |
| OCR | EasyOCR (zh+en) | ~100 MB download | ~500 MB |
| VLM | SmolVLM-256M-Instruct | ~500 MB | ~500 MB |
| **Total** | | **~1.1 GB** | **~1.5 GB (< 2 GB)** |

```

## Hermes Agent Integration

rag-kit includes a SKILL.md that teaches Hermes Agent to use `rag query --json` instead of `web_search` when the user asks about "my documents", "search my data", or "the database".

The installer copies the skill to `~/.hermes/skills/research/rag-kit/SKILL.md`. After installation, Hermes Agent will:
- Detect queries about local documents
- Run `rag query --json --top-k 5 "query"`
- Present results with source attribution

## Testing

```bash
cd rag-kit

# Run the comprehensive pipeline tests (27 tests, no model download needed)
python -m pytest tests/test_rag_pipeline.py -v

# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_cli.py -v
python -m pytest tests/test_embed_store.py -v
python -m pytest tests/test_vlm.py -v        192.168.100.10/24 dev enp1s0f1np1 sudo ip link set enp1s0f1np1 up.

        On Node 2, assign an IP and bring the interface up: sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1 sudo ip link set enp1s0f1np1 up.

        Verify IP assignments on both nodes with ip addr show enp1s0f1np1.

        On Node 1, generate an SSH key pair: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "".

        Copy the public key to Node 2: ssh-copy-id -i ~/.ssh/id_ed25519.pub nvidia@192.168.100.11.

        Test passwordless SSH from Node 1 to Node 2: ssh nvidia@192.168.100.11 hostname.

        Verify connectivity with ping from Node 1 to Node 2: ping -c 4 192.168.100.11 and from Node 2 to Node 1.

    Optional: Persistent Network Configuration

        Create /etc/netplan/99-multinode.yaml on each node with the assigned IP and apply using sudo netplan apply to persist settings across reboots.

    Optional: Bandwidth Test

        On Node 2, run iperf3 -s; on Node 1, run iperf3 -c 192.168.100.11 -t 10 to measure throughput.
    Learn more:
    1 -build.nvidia.com
    2 -deepwiki.com
    3 -github.com
    See less
    Feedback
     
    Global web icon
    nvidia.com
    https://build.nvidia.com › spark › connect-two-sparks
    Connect Two Sparks | DGX Spark

    Nov 24, 2025 · You will physically connect two DGX Spark devices with a QSFP cable, configure network interfaces for cluster communication, and establish passwordless SSH between nodes to …
     
    Global web icon
    deepwiki.com
    https://deepwiki.com › NVIDIA › dgx-spark-playbooks
    Connecting Two Sparks | NVIDIA/dgx-spark-playbooks | DeepWiki

    Mar 23, 2026 · This document covers the physical and network setup required to connect two DGX Spark devices for distributed workloads. This includes QSFP cable connection, network interface …
    Global web icon
    Collabnix
    https://collabnix.com › how-to-connect-two-nvidia-dgx-spark-nodes-as-kubernetes-gpu...
    How to Connect Two NVIDIA DGX Spark Nodes as Kubernetes …

    Jun 21, 2026 · A hot topic in the NVIDIA DGX Spark community is how to connect two Spark nodes for distributed GPU workloads. This guide shows you how to set up two DGX Spark systems as GPU …
    Global web icon
    nvidia.com
    https://build.nvidia.com › spark › connect-two-sparks › stacked-sparks
    Connect Two Sparks | DGX Spark - build.nvidia.com

    Connect the QSFP cable between both DGX Spark systems using any QSFP interface on each device. This establishes the 200GbE direct connection required for high-speed inter-node communication.
    Global web icon
    deepwiki.com
    https://deepwiki.com › NVIDIA › dgx-spark-playbooks
    Multi-Node Setups | NVIDIA/dgx-spark-playbooks | DeepWiki

    Mar 23, 2026 · DGX Spark supports two primary physical connectivity patterns for multi-node clusters: direct back-to-back connection and switched fabric. The following diagram bridges physical hardware …
    Global web icon
    Collabnix
    https://collabnix.com › docker › how-to-connect-two...
    How to Connect Two NVIDIA DGX Spark Nodes as Kubernetes …

    Jun 21, 2026 · A hot topic in the NVIDIA DGX Spark community is how to connect two Spark nodes for distributed GPU workloads. This guide shows you how to set up two DGX Spark systems as GPU …
    Global web icon
    orhanyildirim.us
    https://orhanyildirim.us › blog
    Building a 256GB AI Cluster on My Desk: Connecting Two NVIDIA DGX ...

    Mar 1, 2026 · The author shares their experience building a 256GB AI cluster using two NVIDIA DGX Sparks for distributed LLM inference. They detail the setup process, including the hardware …
    Global web icon
    Improve & Repeat
    https://improveandrepeat.com › how-to-connect-two-nvidia-dgx-sparks
    How to Connect Two NVIDIA DGX Sparks - Improve & Repeat

    6 days ago · The official guide covers all the important points we need to connect two DGX Sparks. However, as so often the tiny little points that are not in the documentation cost a lot of time.
    Global web icon
    Github
    https://github.com › ArgentAIOS › dgx-spark-cluster
    ArgentAIOS/dgx-spark-cluster - GitHub

    Apr 10, 2026 · Complete setup guide for a 2-node NVIDIA DGX Spark cluster — distributed training, CUDA inference with EXO, NCCL tuning for Grace Blackwell, NVMe-TCP shared storage, and 200 …
    People also ask
     
    Some results are removed in response to a notice of local law requirement. For more information, please see here.
    Some results have been hidden because they may be inaccessible to you.
    Show inaccessible results
        1
        2
        3


```

## Troubleshooting

### "ModuleNotFoundError" or import errors
```bash
# Ensure you're in the right venv
source ~/rag-kit-venv/bin/activate       # Linux
~\rag-kit-venv\Scripts\activate.bat      # Windows
```

### "HF_HUB_OFFLINE=1 but model not cached"
```bash
# Pre-download models first
scripts/download-models.bat   # Windows
# or
# Set China mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### "Permission denied" on setup-autostart (Windows)
Run as Administrator or skip autostart — the watcher can be started manually:
```bash
rag watch ~\Documents\rag-ingest
```

### Watcher not running
```bash
# Check watcher status
rag status

# Start manually
rag watch ~/Documents/rag-ingest &

# Check autostart
rag setup-autostart --json
```

### Empty query results
```bash
# Check if documents were ingested
rag list-files --json

# Re-ingest
rag ingest ~/Documents/rag-ingest
```

## Requirements

- Python 3.10+
- 4 GB RAM recommended (2 GB minimum for models)
- ~2 GB disk for models + documents
- NVIDIA GPU optional (CPU-only supported)

## Platforms

| Platform | Install Script | Autostart |
|---|---|---|
| Windows 10/11 | `scripts\install-windows.bat` | schtasks |
| Linux (x86_64) | `scripts/install-dgx-spark.sh` | systemd user service |
| Linux (DGX Spark ARM64) | `scripts/install-dgx-spark.sh` | systemd user service |

## License

MIT
