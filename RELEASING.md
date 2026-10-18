# Releasing

Publishing to PyPI is done by the `release` workflow when a version tag is pushed. It does not
run on ordinary pushes, and nothing is published until the steps below are done by a maintainer.

One-time setup (needs the maintainer's PyPI account):

1. On PyPI, create the `isbetter` project or a *pending publisher* for it, with this repository,
   workflow `release.yml` and environment `pypi` (trusted publishing: no token is stored).
2. In the GitHub repository settings, create an environment named `pypi`. Requiring a reviewer
   there adds a manual approval before anything is uploaded.

For each release:

1. Update the version in `pyproject.toml` and `src/isbetter/__init__.py` (a test keeps the two
   equal) and the date and entries in `CHANGELOG.md`.
2. Merge to `main`, then tag: `git tag v0.1.0 && git push origin v0.1.0`.
3. The workflow runs lint, types and tests, checks the tag matches the version, builds the sdist
   and wheel, runs `twine check`, installs the wheel into a clean environment and runs the CLI on
   an example, and only then publishes.

Dry run without publishing: run the workflow from the Actions tab (`workflow_dispatch`); the
publish job runs only for tags. Locally: `uv build && uvx twine check dist/*`.
