import merge from 'package-merge';
import fs from 'fs';
import path from 'path';

const getPackage = () => {
  let origPackage;
  try {
    origPackage = fs.readFileSync(path.join(process.cwd(), 'package.json'), 'utf-8');
  } catch (e) {
    origPackage = false;
  }
  return origPackage;
};

const mergePackages = (origPackage, destinations) => compiledFiles => {
  if (origPackage) {
    const newPackagePath = path.join(process.cwd(), 'package.json');
    const packageIndex = destinations.indexOf(newPackagePath);
    const packageContents = compiledFiles[packageIndex];
    const mergedPack = merge(origPackage, packageContents);
    compiledFiles[packageIndex] = mergedPack;
  }
  return compiledFiles;
};
export {getPackage, mergePackages};
