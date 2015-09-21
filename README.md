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

```sh
npm install -g cf-package
mkdir project
cd project
cf-package
npm install
npm test
```

### Input via propmt:

```sh
cf-package
> Your name: My Name
> Your email (will be publicly available, optional): my@email 
> Your GitHub public username: mygithub
> Package name: test
> Package description: Test Package
```

Variable            | Prompt
------------------- | ---
user.name           | > Your name:
user.email          | > Your email (will be publicly available, optional):
user.github         | > Your GitHub public username:
package.name        | > Package name:
package.description | > Package description:

### Quick input

Alternatively it is possible to input every generator variable as CLI option:

```sh
cf-package --user.name="My Name" \
  --user.email=my@email \
  --user.github=mygithub \
  --package.name=test \
  --package.description="Test Package"
```

Explore and enjoy!  Part of the [Cloverfield project](https://github.com/cloverfield-tools/cloverfield).
