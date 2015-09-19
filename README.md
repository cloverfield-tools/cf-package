# cf-package [![Circle CI](https://circleci.com/gh/cloverfield-tools/cf-package/tree/master.svg?style=svg)](https://circleci.com/gh/cloverfield-tools/cf-package/tree/master)

Cloverfield Package Scaffold

## What does this do?

It scaffolds a new module with the following features:

* ES6 with Babel
* Lint with ESLint
* Tape tests with coverage report
* Dependency security audits with nsp
* Ensure dependencies are properly declared in package.json
* Git precommit hook enforces quality checks on commit
* CI config (Travis, CircleCI)


## Getting Started

```
npm install -g cf-package
mkdir project
cd project
cf-package
npm install
npm test
```

Explore and enjoy!  Part of the [Cloverfield project](https://github.com/cloverfield-tools/cloverfield).
