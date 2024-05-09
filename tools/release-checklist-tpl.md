# Release Issue Checklist

Copy the template below the line, substitute (`s/<VERSION>/1.2.3/`) the correct
version and create an [issue](https://github.com/has2k1/mizani/issues/new).

The first line is the title of the issue

------------------------------------------------------------------------------
Release: mizani-<VERSION>

- [ ] Run tests and coverage locally

  ```sh
  git switch main
  git pull origin/main
  make test
  make coverage
  ```

  - [ ] The tests pass
  - [ ] The coverage is acceptable

- [ ] The latest [online documentation](https://mizani.readthedocs.io/en/latest/) builds, be sure to browse

- [ ] Create a release branch

  ```sh
  git switch -c release-v<VERSION>
  ```

- [ ] Tag a pre-release version. These are automatically deployed on `testpypi`

  ```sh
  git tag -as v<VERSION>a1 -m "Version <VERSION>a1"  # e.g. <VERSION>a1, <VERSION>b1, <VERSION>rc1
  git push -u origin release-v<VERSION>
  ```
  - [ ] GHA [release job](https://github.com/has2k1/mizani/actions/workflows/release.yml) passes
  - [ ] Mizani test release is on [TestPyPi](https://test.pypi.org/project/mizani)


- [ ] Update changelog

  ```sh
  nvim doc/changelog.rst
  git commit -am "Update changelog for release"
  git push
  ```
  - [ ] Update / confirm the version to be released
  - [ ] Add a release date
  - [ ] The [GHA tests](https://github.com/has2k1/mizani/actions/workflows/testing.yml) pass

- [ ] Tag final version and release

  ```sh
  git tag -as v<VERSION> -m "Version <VERSION>"
  git push
  ```

  - [ ] The [GHA Release](https://github.com/has2k1/mizani/actions/workflows/release.yml) job passes
  - [ ] [PyPi](https://pypi.org/project/mizani) shows the new release

- [ ] Update `main` branch

  ```sh
  git switch main
  git merge --ff-only release-v<VERSION>
  git push
  ```

- [ ] Create conda release

  - [ ] Copy _SHA256 hash_. Click view hashes, for the [Source Distribution](https://pypi.org/project/mizani/<VERSION>/#files) (`.tar.gz`).

  - [ ] Update [mizani-feedsock](https://github.com/conda-forge/mizani-feedstock)

    ```sh
    cd ../mizani-feedstock
    git switch main
    git pull upstream main
    git switch -c v<VERSION>
    nvim recipe/meta.yaml
    git commit -am  "Version <VERSION>"
    git push -u origin v<VERSION>
    ```
  - [ ] Create a [PR](https://github.com/conda-forge/mizani-feedstock/pulls)
  - [ ] Complete PR (follow the steps and merge)

- [ ] Add [zenodo badge](https://doi.org/10.5281/zenodo.592370) to the changelog.
