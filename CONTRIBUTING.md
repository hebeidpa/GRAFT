# Contributing

Do not commit patient data, real NIfTI files, credentials, local absolute paths,
or institution-only model files. Use synthetic fixtures for tests.

Before opening a pull request:

```bash
python -m unittest discover -s tests -v
git status --short
git diff --cached
```

Any change to the fixed cohort schema must update the loader, template, README,
tests, and released checkpoint compatibility notes together.
