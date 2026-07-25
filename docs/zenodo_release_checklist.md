# Zenodo and Journal Release Checklist

Use this checklist before creating a release tag and Zenodo archive.

## Metadata

- [ ] README.md is up to date.
- [ ] CITATION.cff contains current version and author information.
- [ ] LICENSE contains full MIT text.
- [ ] CHANGELOG.md has release notes for this version.

## Reproducibility

- [ ] environment.yml and requirements.txt are current.
- [ ] pyproject.toml installs correctly with pip install -e .
- [ ] Tests pass with pytest -q.
- [ ] Reproducibility commands in docs/reproducibility_guide.md are verified.

## Manuscript assets

- [ ] Figure 1 schematic regenerated.
- [ ] Behavioural figures regenerated.
- [ ] Neural analysis and neural figures regenerated.
- [ ] Manuscript tables regenerated.
- [ ] Main text figure count is 8 or fewer.

## Repository hygiene

- [ ] No build artifacts or local caches tracked.
- [ ] No secrets or local machine paths committed.
- [ ] Large binary artifacts are intentionally versioned or externally archived.

## Release process

- [ ] Create an annotated git tag (for example, v1.0.0).
- [ ] Create GitHub release notes.
- [ ] Archive release in Zenodo and mint DOI.
- [ ] Add DOI back into CITATION.cff and README.md.
