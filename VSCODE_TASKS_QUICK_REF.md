# VS Code Tasks - Quick Reference Card

Press `Ctrl+Shift+P` → `Tasks: Run Task` (or `Ctrl+Shift+B`)

## 🚀 Most Common Tasks

| Task | Purpose | When to Use |
|------|---------|-------------|
| **Streamlit: Run Assessment App** | Launch interactive assessment system | Taking assessments, viewing progress |
| **Jupyter: Start Lab** | Launch JupyterLab | Working with notebooks |
| **Env: Setup Full** | Install all dependencies | First-time setup |
| **Assessment: Validate All** | Check all assessment files | After editing assessments |
| **Database: Backup Assessment Results** | Save progress data | Before major changes |

---

## 📋 Environment Setup

```
Env: Create Virtual Environment          → Create .venv folder
Env: Activate Virtual Environment (Info) → Show activation commands
Env: Setup Basic                         → Install core libraries
Env: Setup Intermediate                  → Add time series tools
Env: Setup Advanced                      → Add deep learning
Env: Setup Full                          → Install everything
Env: Recreate Full (Force)               → Fresh environment
Env: Install Streamlit Requirements      → Streamlit only
Env: Install All Requirements            → Direct pip install
Env: Upgrade pip                         → Update pip
```

---

## 📱 Streamlit App

```
Streamlit: Run Assessment App                      → Launch app (http://localhost:8501)
Streamlit: Run Assessment App (with auto-reload)   → Development mode
Streamlit: Clear Cache and Run                     → Fresh start
```

**Stop App**: Press `Ctrl+C` in terminal panel

---

## 🗄️ Database

```
Database: Clear Assessment Results    → Delete all progress ⚠️
Database: Backup Assessment Results   → Create timestamped backup
```

---

## ✅ Validation

```
Assessment: Validate All              → Check all modules
Assessment: Validate Specific Module  → Check one module (pick from list)
```

---

## 🧪 Testing & Jupyter

```
Jupyter: Start Lab                    → Launch JupyterLab
Jupyter: Start Notebook               → Launch classic Jupyter Notebook
Tests: Run All                        → Run all tests with pytest
Tests: Run with Coverage              → Run tests with coverage report
```

---

## 🎯 Quick Workflows

### First Time User

1. `Env: Create Virtual Environment`
2. Activate environment in terminal
3. `Env: Install Streamlit Requirements`
4. `Streamlit: Run Assessment App`

### Developer

1. Activate `.venv`
2. `Streamlit: Run Assessment App (with auto-reload)`
3. Edit files → auto-reload
4. `Assessment: Validate All`

### Fresh Start

1. `Database: Backup Assessment Results`
2. `Env: Recreate Full (Force)`
3. `Streamlit: Clear Cache and Run`

---

## 💡 Pro Tips

- **Background tasks**: Stop with `Ctrl+C` in terminal
- **Keyboard shortcut**: `Ctrl+Shift+B` opens task menu
- **Rerun last task**: `Ctrl+Shift+P` → "Tasks: Rerun Last Task"
- **Environment**: Always activate `.venv` before manual commands

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Task not found | Reload window: `Ctrl+Shift+P` → "Developer: Reload Window" |
| streamlit: command not found | Run `Env: Install Streamlit Requirements` |
| Database locked | Close all Streamlit instances, then clear database |
| Permission denied (Windows) | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

**Full documentation**: [.vscode/TASKS_README.md](.vscode/TASKS_README.md)
