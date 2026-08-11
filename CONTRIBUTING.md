# Contributing

Contributions are welcome, though please for opinionated changes discuss
them first in an issue before making a pull request.

## Developer instructions

We use Pixi for our development workflow (a modern alternative to Conda).

- [install Pixi](https://pixi.prefix.dev/latest/installation/)
- fork and clone the repository, and `cd` into the repository
- run `pixi task list` (which will show you the available commands for the repository - such as building the documentation, or running tests)
- run the command you would like using `pixi run ...` (e.g, `pixi run tests`)

You can also enter the environment for the repository by doing `pixi shell` (this is equivalent to `conda activate ...`).

The installation of the virtual environment will be done automatically by Pixi when running `pixi run` or `pixi shell` - this environment is visible in the `./pixi` folder under the repo root.

The `.pixi` folder and the `pixi.lock` file should not be editted by hand. The `pixi.lock` file should be committed.


## Maintainer instructions

### About the `pixi.lock`

This repo has a committed `pixi.lock` (meaning that the tree of dependencies used in development is static).
This results in perfectly reproducibility between dev machines and CI.

This, however, means that releases of dependencies (e.g., deprecations in Parcels or Xarray) may unknowingly break Plasticparcels.
To mitigate this - there is a Renovate Bot (i.e., an alternative to Dependabot) that will upgrade the Pixi lock on a regular basis and automerge if CI passes.
If CI fails, there is a change upstream that Plasticparcels needs a code adjustment to deal with.

### Release checklist

- Make sure CI is passing on the latest version of `main`
- Go to GitHub, draft new release. Enter name of version and "create new tag" if it doesn't already exist. Click "Generate Release Notes". Currate release notes as needed. Look at a previous version release to match the format (title, header, section organisation etc.)
- Go to [conda-forge/plasticparcels-feedstock](https://github.com/conda-forge/plasticparcels-feedstock), create a new issue (select the "Bot Commands" issue from the menu) with title `@conda-forge-admin, please update version`. This will prompt a build, otherwise there can be a delay in the build.
  - Approve PR and merge on green