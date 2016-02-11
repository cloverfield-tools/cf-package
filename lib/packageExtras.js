// import fs from 'fs';
// import path from 'path';
// const templatePackage = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'template', 'package.json'), 'utf-8'));
// const templatePackage = JSON.parse(fs.readFileSync(path.join('..', '/template', 'package.json'), 'utf-8'));
const templatePackage = require('../template/package.json');

const packageExtras = userPackage => props => {
  if (props['package.dependencies']) {
    props['package.dependencies'] = {...templatePackage.dependencies, ...userPackage.dependencies};
  } else {
    props['package.dependencies'] = {...templatePackage.dependencies};
  }
  if (props['package.devDependencies']) {
    props['package.devDependencies'] = {...templatePackage.devDependencies, ...userPackage.devDependencies};
  } else {
    props['package.devDependencies'] = {...templatePackage.devDependencies};
  }
  if (props['package.scripts']) {
    props['package.scripts'] = {...templatePackage.scripts, ...userPackage.scripts};
  } else {
    props['package.scripts'] = {...templatePackage.scripts};
  }
  return props;
}

export default packageExtras;
