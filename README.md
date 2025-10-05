# drug-prio-eval
## Environment Variables
- Put environment variables in `.env` file.
- Usage in Python:
  ```python
  import os
  from dotenv import load_dotenv

  load_dotenv(find_dotenv())  # Load environment variables from .env file

  # Accessing an environment variable
  db_url = os.getenv('DATABASE_URL')
  ```
## Naming & Formatting Conventions

### 📁 Directories & Files
- **Lowercase snake_case** for all directories and filenames:
  - Nextflow: `main_workflow.nf`
  - Python: `data_utils.py`
  - Shell: `run_pipeline.sh`
  - Docker context: `docker/webapp/`
- Dockerfiles located in `docker/` named `Dockerfile` or `Dockerfile.<service>`  
- Docker Compose file: `docker-compose.yml`

### 🧩 Nextflow
- **Process names**: PascalCase (e.g. `ProcessAlignReads`)  
- **Channels & params**: snake_case (e.g. `raw_reads_ch`)

### 🐍 Python
- **Modules/Packages**: snake_case (e.g. `file_parser.py`)  
- **Functions & variables**: snake_case (e.g. `load_data()`, `sample_count`)  
- **Classes**: PascalCase (e.g. `DataLoader`)  
- **Constants**: UPPER_SNAKE_CASE (e.g. `MAX_RETRIES`)

### 🐚 Shell
- **Functions**: snake_case (e.g. `process_files()`)  
- **Local variables**: snake_case (e.g. `output_dir`)  
- **Env vars & constants**: UPPER_SNAKE_CASE (e.g. `LOG_LEVEL`)

### 🐳 Docker
- **Image names & tags**: lowercase, hyphens, version after `:` (e.g. `myorg/my-image:1.0.0`)  
- **Compose services**: snake_case (e.g. `db_service`)  
- **Container names**: lowercase with underscores/hyphens (e.g. `web_app_1`)

## Environment Variables

Before running the application, copy the example file and populate it with your own credentials:

```bash
cp .env.example .env
