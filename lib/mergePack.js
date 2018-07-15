import fs from 'fs';
import path from 'path';
import merge from 'lodash.merge';
import parseAuthor from 'parse-author';
import parseGithub from 'parse-github-url';

export const getPackage = () => {
  let origPackage;
  try {
    origPackage = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'package.json'), 'utf-8'));
  } catch (e) {
    origPackage = false;
  }
  return origPackage;
};

export const getPackageProps = (existingPackage, packageProperties, schema) => {
  let gitHub, authorInfo;
  if (existingPackage) {
    console.log(`Existing package.json file has been used to get default variables.`);
    schema.properties = {...schema.properties, ...packageProperties};
    schema.properties['package.name'].default = existingPackage.name;
    schema.properties['package.description'].default = existingPackage.description;
    // Try to get Name, Email & GitHub Username
    if (existingPackage.repository && existingPackage.repository.url) {
      gitHub = parseGithub(existingPackage.repository.url).owner;
      schema.properties['user.github'].default = gitHub;
    }
    if (existingPackage.author) {
      authorInfo = parseAuthor(existingPackage.author);
      if (authorInfo.name) {
        schema.properties['user.name'].default = authorInfo.name;
      }
      if (authorInfo.email) {
        schema.properties['user.email'].default = authorInfo.email;
      }
    }
  }
  return schema;
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
