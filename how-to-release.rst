##############
How to release
##############

Testing
=======

* `cd` to the root of project and run
  ::

    make test

* Or, to test in all environments
  ::

    tox

* Once all the tests pass move on





Tagging
=======

Check out the master branch, edit the changelog and add the tag
release date, tag with the version number & push the tags

  ::

    git checkout master
    vim doc/changelog.rst
    git add doc/changelog.rst
    git commit -m 'Prepare changelog for release'
    git tag -a v0.1.0 -m 'Version: 0.1.0'
    git push upstream --tags

Note the `v` before the version number.


Release
=======

* Make sure your `.pypirc` file is setup
  `correctly <http://docs.python.org/2/distutils/packageindex.html>`_.
  ::

    cat ~/.pypirc

* Release

 ::

    make release

* Draft a `github release <https://github.com/has2k1/mizani/releases>`.

* Add `zenodo badge <https://zenodo.org/account/settings/github/repository/has2k1/mizani>`
  to the changelog.

* Add a new version (unreleased) section at the top of the changelog.

* Commit ::

    git add doc/changelog.rst
    git commit -m 'Add zenodo badge for previous release'

* Create conda release ::

    sha256sum dist/*.tar.gz
    # edit mizani-feedsock/recipe/meta.yml
    # push feedstock changes and create a PR at conda-forge
    # Wait for tests to pass and merge PR


