# Release Issue Checklist

Copy the template below the line, substitute (`s/<VERSION>/1.2.3/`) the correct
version and create an [issue](https://github.com/has2k1/mizani/issues/new).

The first line is the title of the issue

------------------------------------------------------------------------------
Release: mizani-<VERSION>

- [ ] Run tests and coverage locally

  <details>

  ```sh
  git switch main
  git pull origin/main
  make test
  make coverage
  ```
  - [ ] The tests pass
  - [ ] The coverage is acceptable

  </details>

- [ ] The latest [online documentation](https://mizani.readthedocs.io/latest/) builds, be sure to browse

- [ ] Create a release branch

  <details>

  ```sh
  git switch -c release-v<VERSION>
  ```

  </details>

- [ ] Tag a pre-release version. These are automatically deployed on `testpypi`

  <details>

  ```sh
  git tag -as v<VERSION>a1 -m "Version <VERSION>a1"  # e.g. <VERSION>a1, <VERSION>b1, <VERSION>rc1
  git push -u origin release-v<VERSION>
  ```
  - [ ] GHA [release job](https://github.com/has2k1/mizani/actions/workflows/release.yml) passes
  - [ ] Mizani test release is on [TestPyPi](https://test.pypi.org/project/mizani)

  </details>


- [ ] Update changelog

  <details>

  ```sh
  nvim doc/changelog.rst
  git commit -am "Update changelog for release"
  git push
  ```
  - [ ] Update / confirm the version to be released
  - [ ] Add a release date
  - [ ] The [GHA tests](https://github.com/has2k1/mizani/actions/workflows/testing.yml) pass

  </details>

- [ ] Tag final version and release

  <details>

  ```sh
  git tag -as v<VERSION> -m "Version <VERSION>"
  git push
  ```

  - [ ] The [GHA Release](https://github.com/has2k1/mizani/actions/workflows/release.yml) job passes
  - [ ] [PyPi](https://pypi.org/project/mizani) shows the new release

  </details>

- [ ] Update `main` branch

  <details>

  ```sh
  git switch main
  git merge --ff-only release-v<VERSION>
  git push
  ```

  - [ ] The [GHA Release](https://github.com/has2k1/mizani/actions/workflows/release.yml) job passes
  - [ ] [PyPi](https://pypi.org/project/mizani) shows the new release

  </details>

- [ ] Create conda release

  <details>


  - [ ] Copy _SHA256 hash_. Click view hashes, for the [Source Distribution](https://pypi.org/project/mizani/<VERSION>/#files) (`.tar.gz`).

  - [ ] Update [mizani-feedsock](https://github.com/conda-forge/mizani-feedstock)

    ```sh
    cd ../mizani-feestock
    git switch main
    git pull upstream main
    git switch -c v<VERSION>
    nvim recipe/meta.yml
    git commit -am  "Version <VERSION>"
    git push -u origin v<VERSION>
    ```
  - [ ] Create a [PR](https://github.com/conda-forge/mizani-feedstock/pulls)
  - [ ] Complete PR (follow the steps and merge)

  </details>

- [ ] Add [zenodo badge](https://zenodo.org/account/settings/github/repository/has2k1/mizani) to the changelog.
