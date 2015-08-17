# {{package.name}} [![Circle CI](https://circleci.com/gh/{{user.github}}/{{package.name}}/tree/master.svg?style=svg)](https://circleci.com/gh/{{user.github}}/{{package.name}}/tree/master)
[![Travis-CI](https://travis-ci.org/{{user.github}}/{{package.name}}.svg)](https://travis-ci.org/{{user.github}}/{{package.name}})


An npm `scripts` boilerplate for modules intended for production.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Clone the repo](#clone-the-repo)
  - [Configure your CI build](#configure-your-ci-build)
  - [Build your module](#build-your-module)
- [Contributing](#contributing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Features

* ES6 with Babel
* Lint with ESLint
* Tape tests with coverage report
* Dependency security audits with nsp
* Ensure dependencies are properly declared in package.json
* Git pre-commit hook enforces quality checks on commit
* Travis CI integration


## Getting Started

### Clone the repo

In a bash terminal:

```sh
git clone git@github.com:{{user.name}}/{{package.name}}.git
cd {{package.name}}
npm install
```

### Configure your CI build

1. Travis CI

  This package comes with a `.travis.yml` file. You need to activate [Travis CI for your repository](http://docs.travis-ci.com/user/getting-started/).

2. Circle CI

  This package comes with a `circle.yml` file. You need to activate [Circle CI for your repository](https://circleci.com/docs/getting-started/).


### Build your module

1. For production

  ```sh
  npm run build
  ```

  It will run webpack once building full and minified versions of your library in `./build` with source-maps.


2. For development

  ```sh
  npm start
  ```

  This will run the `webpack` build in watch mode and will include ESLint checks on compile time.

  ![webpack](https://cloud.githubusercontent.com/assets/175264/8304834/d66f7944-19ec-11e5-9feb-9f66caa5c593.gif)

  **Note** minified version will not be built in dev mode.

## Contributing

- [Contributing](docs/contributing/index.md)
  - [Versions: Release Names vs Version Numbers](docs/contributing/versions/index.md)
