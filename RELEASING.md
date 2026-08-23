# Releasing Lodstone

## One-time PyPI setup

1. Create a pending trusted publisher for the `lodstone` project on PyPI.
2. Set the owner to `kephale`, repository to `lodstone`, workflow to
   `release.yml`, and environment to `pypi`.
3. Create a protected `pypi` environment in the GitHub repository. Requiring a
   reviewer is recommended.

No PyPI token is stored in GitHub. The release workflow uses OpenID Connect
trusted publishing and uploads PyPI attestations.

## Release checklist

1. Ensure `CHANGELOG.md` describes the release and has a release date.
2. Set the version in `pyproject.toml` and refresh `uv.lock`.
3. Run the complete local gate:

   ```bash
   uv run --group dev ruff check .
   uv run --group dev ruff format --check .
   uv run --extra ome-zarr --group dev pyright src
   uv run --extra test pytest
   uv build
   uvx twine check dist/*
   ```

4. Merge the release commit to `main` and wait for CI.
5. Create a GitHub release whose tag is exactly `v<package-version>`.
6. Verify the `release` workflow and the files and attestations on PyPI.
7. Install the release into a clean environment and run the README example.
8. Update the compatible napari, ndv, and ChimeraX integrations to the released
   version and run their native smoke tests.

The workflow refuses to publish when the Git tag and package version differ.
