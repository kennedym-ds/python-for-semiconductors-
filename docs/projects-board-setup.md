## Projects Board Setup

Follow these steps to create a GitHub Projects (v2) board and enable auto-triage from issues/PRs.

### 1) Create the Project Board
- Create a new GitHub Projects (v2) board at the user or org level.
- Copy the board URL, e.g., `https://github.com/users/<owner>/projects/<number>` or `https://github.com/orgs/<org>/projects/<number>`.

### 2) Create a Fine-Grained PAT
- Create a fine-grained Personal Access Token with:
  - Repository permissions: Issues (Read/Write), Pull Requests (Read/Write), Contents (Read)
  - Organization permissions (if org project): Projects (Read/Write)
- Name the secret `PROJECTS_TOKEN` in the repository Settings → Secrets and variables → Actions.

### 3) Add the Project URL Secret
- Add a secret named `PROJECTS_BOARD_URL` with the full Projects board URL.

### 4) Optional: Project Field Mapping
- Create a text field named `Category` in your Project.
- The workflow will map labels (e.g., `starter`, `advanced`, `classification`, `regression`, `time-series`, `computer-vision`, `mlops`) into this field when present.

### 5) Labels Recommendation
- Use consistent labels for auto-triage:
  - `projects`, `starter`, `advanced`, `classification`, `regression`, `time-series`, `computer-vision`, `mlops`

### 6) Verify
- Open or label an issue; it should appear on the Projects board automatically.
