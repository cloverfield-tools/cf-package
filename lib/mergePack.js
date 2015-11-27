import merge from 'lodash.merge';
import fs from 'fs';
import path from 'path';

export const getPackage = () => {
  let origPackage;
  try {
    origPackage = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'package.json'), 'utf-8'));
  } catch (e) {
    origPackage = false;
  }
  return origPackage;
};

export const mergePackages = (origPackage, destinations) => compiledFiles => {
  if (origPackage) {
    const newPackagePath = path.join(process.cwd(), 'package.json');
    const packageIndex = destinations.indexOf(newPackagePath);
    const packageContents = JSON.parse(compiledFiles[packageIndex]);
    const mergedPack = JSON.stringify(merge(packageContents, origPackage), null, 2);
    compiledFiles[packageIndex] = mergedPack;
  }
  return compiledFiles;
};
