import templatePackage from '../template/package.json';

const packageFields = ['dependencies', 'devDependencies', 'scripts'];

const setPackageProps = userPackage => props => {
  packageFields.forEach((field) => {
    if (props[`package.${field}`]) {
      props[`package.${field}`] = {...templatePackage[field], ...userPackage[field]};
    } else {
      props[`package.${field}`] = {...templatePackage[field]};
    }
  });
  return props;
};

export default setPackageProps;
