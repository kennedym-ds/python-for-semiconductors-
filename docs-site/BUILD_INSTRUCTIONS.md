# Documentation Site Build Instructions

This document provides instructions for building and maintaining the documentation site for the Python for Semiconductors learning series.

## Prerequisites

Ensure you have Python 3.8+ and the advanced-tier dependencies installed:

```bash
# Install advanced tier dependencies (includes mkdocs)
python env_setup.py --tier advanced

# Or install mkdocs directly
pip install mkdocs mkdocs-material
```

## Quick Start

### 1. Set up Documentation Site

Initialize the mkdocs project structure:

```bash
cd /path/to/python-for-semiconductors/
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  setup-mkdocs --project-dir ./docs-site
```

This creates:
- `docs-site/mkdocs.yml` - MkDocs configuration
- `docs-site/docs/` - Documentation source directory
- `docs-site/docs/index.md` - Homepage content

### 2. Build Documentation Site

Generate the static documentation site:

```bash
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  build-docs --project-dir ./docs-site
```

Or use mkdocs directly:

```bash
cd docs-site
mkdocs build
```

Built site will be available in `docs-site/site/`

### 3. Serve Locally for Development

Start a local development server with auto-reload:

```bash
cd docs-site
mkdocs serve
```

Site will be available at http://localhost:8000

## Documentation Workflow

### Convert Notebooks to Documentation

Convert Jupyter notebooks to markdown for inclusion in the site:

```bash
# Convert specific module notebooks
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  generate-docs \
  --input modules/foundation/module-3/ \
  --output docs-site/docs/foundation/module-3/ \
  --format markdown

# Convert to both markdown and HTML
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  generate-docs \
  --input modules/foundation/module-3/ \
  --output docs-site/docs/foundation/module-3/ \
  --format both
```

### Validate Dataset Paths

Ensure all notebooks follow standard dataset path patterns:

```bash
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  validate-paths --modules-dir modules/
```

### Export Environment Specifications

Create reproducible environment files:

```bash
# Export conda environment
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  export-env --output environment.yml --format conda

# Export pip requirements
python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
  export-env --output requirements.txt --format pip
```

## Configuration

### MkDocs Configuration (`mkdocs.yml`)

The generated configuration includes:

```yaml
site_name: Python for Semiconductors Documentation
site_description: ML Learning Series for Semiconductor Engineers

theme:
  name: material
  palette:
    primary: blue
    accent: light-blue
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.suggest

nav:
  - Home: index.md
  - Foundation:
    - Module 1: foundation/module-1.md
    - Module 2: foundation/module-2.md
    - Module 3: foundation/module-3.md
  - Intermediate:
    - Module 4: intermediate/module-4.md
  - Project Development:
    - Module 10: project-dev/module-10.md

markdown_extensions:
  - codehilite
  - admonition
  - toc
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed
```

### Customization

To customize the documentation:

1. **Add new pages**: Create `.md` files in `docs-site/docs/` and update the `nav` section in `mkdocs.yml`
2. **Change theme**: Modify the `theme` section in `mkdocs.yml`
3. **Add extensions**: Add to the `markdown_extensions` list

## Deployment

### GitHub Pages

Deploy to GitHub Pages using mkdocs:

```bash
cd docs-site
mkdocs gh-deploy
```

This will:
1. Build the documentation
2. Push to the `gh-pages` branch
3. Enable GitHub Pages hosting

### Manual Deployment

For manual deployment, build the site and copy the `site/` directory to your web server:

```bash
mkdocs build
# Copy docs-site/site/* to your web server
```

## CI/CD Integration

### GitHub Actions

Example workflow for automated documentation builds:

```yaml
name: Documentation
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements-advanced.txt

      - name: Validate paths
        run: |
          python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
            validate-paths --modules-dir modules/

      - name: Generate docs
        run: |
          python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
            generate-docs --input modules/ --output docs-site/docs/ --format markdown

      - name: Build site
        run: |
          python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py \
            build-docs --project-dir docs-site/

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        run: |
          cd docs-site
          mkdocs gh-deploy --force
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: validate-paths
        name: Validate dataset paths
        entry: python modules/project-dev/module-10/10.3-documentation-reproducibility-pipeline.py validate-paths --modules-dir modules/
        language: python
        pass_filenames: false
```

## Troubleshooting

### Common Issues

**Issue**: `mkdocs: command not found`
```bash
# Solution: Install mkdocs
pip install mkdocs mkdocs-material
```

**Issue**: Theme not loading correctly
```bash
# Solution: Ensure mkdocs-material is installed
pip install mkdocs-material
```

**Issue**: Navigation not working
- Check that all referenced files exist in the `docs/` directory
- Verify paths in the `nav` section of `mkdocs.yml`

**Issue**: Search not working
- Search is enabled by default with Material theme
- Ensure JavaScript is enabled in your browser

### Performance Tips

- Use `--format markdown` instead of `both` for faster conversion
- Exclude large binary files from documentation directories
- Use `.gitignore` to exclude generated `site/` directory from version control

## File Structure

After setup, your documentation structure will be:

```
docs-site/
├── mkdocs.yml              # Configuration file
├── docs/                   # Source documentation
│   ├── index.md           # Homepage
│   ├── foundation/        # Foundation modules docs
│   ├── intermediate/      # Intermediate modules docs
│   └── project-dev/       # Project development docs
└── site/                  # Generated site (after build)
    ├── index.html
    ├── assets/
    └── ...
```

## Getting Help

- MkDocs Documentation: https://www.mkdocs.org/
- Material Theme Documentation: https://squidfunk.github.io/mkdocs-material/
- Pipeline CLI Help: `python 10.3-documentation-reproducibility-pipeline.py --help`
- Subcommand Help: `python 10.3-documentation-reproducibility-pipeline.py <command> --help`
