## 🔧 1. **Install pre-commit**

In your terminal:

```bash
pip install pre-commit
```

Then in your repo root:

```bash
pre-commit install
```

This installs the Git hook so it runs automatically before every commit.

---

## 🧰 3. **Setup & Dependencies**

If it’s Python:

- Use `requirements.txt` **or** `pyproject.toml` (with Poetry/pip).
- Consider making it installable via `setup.py` if you want it to be used as a library.
- Add a `.gitignore` (use [gitignore.io](https://www.toptal.com/developers/gitignore) for a template).

---

## 🔍 4. **Documentation**

You don’t need a full-blown website, but some docs help:

- Put a `docs/` folder with markdown files if needed.
- You can later generate a GitHub Pages site using [MkDocs](https://www.mkdocs.org/) or [Sphinx](https://www.sphinx-doc.org/).

---

## 🧪 5. **Testing**

Include a `tests/` directory. Even a few basic unit tests can go a long way in showing that the code works.

Use:

```bash
pytest tests/
```
