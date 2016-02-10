#! /usr/bin/env node


import 'babel-polyfill';
import prompt from 'prompt';
import {parse} from 'nomnom';
import glob from 'glob';
import parseProps from './parseProps';
import generate from './generate';
import {getPackage, mergePackages} from './mergePack';
import fs from 'fs';
import mkdirp from 'mkdirp';
import path from 'path';
import {packageProperties, properties, defaults} from './properties';
import {
  blackBright as grey,
  greenBright as green,
  yellowBright as yellow,
  redBright as red} from 'cli-color';

let schema = {properties};

let gitHub, author;
const existingPackage = getPackage();
if (existingPackage) {
  console.log(`Existing package.json file has been used to get default variables.`);
  // console.log('name is ', existingPackage.name);
  schema.properties = {...schema.properties, ...packageProperties};
  schema.properties['package.name'].default = existingPackage.name;
  schema.properties['package.description'].default = existingPackage.description;
  if (existingPackage.repository) {
    gitHub = existingPackage.repository.url.match(/github.com:(.*)\//);
    if (gitHub) {
      schema.properties['user.github'].default = gitHub[1];
    }
  }
  if (existingPackage.author) {
    schema.properties['user.name'].default = existingPackage.author;
    // author = existingPackage.author.match(/(.*)((?:\s<)([\w\d@\.]*)(?:>)){0,1}/);
    // if (author) {
    //   schema.properties['user.name'].default = author[1];
    //   if (author.length == 3) {
    //     schema.properties['user.email'].default = author[2];
    //   }
    // }
  }
  console.dir(schema);
}


Object.assign(prompt, {
  message: '>'.green,
  delimiter: ' ',
  colors: false
});
prompt.start();


const sources = glob.sync('../template/**/*', {realpath: true, nodir: true, dot: true, cwd: __dirname});
const destinations = sources.map(source =>
  path.join(process.cwd(), path.relative(path.join(__dirname, '..', 'template'), source)));
const srcContent = sources.map(fileName => fs.readFileSync(fileName, 'utf-8'));


const getPrompt = () => new Promise((resolve, reject) =>
  prompt.get(schema, (err, props) => err ? reject(err) : resolve(props)));


const saveCompiled = compiledFiles => compiledFiles.forEach((content, key) => {
  mkdirp.sync(path.dirname(destinations[key]));

  const from = path.relative(path.join(__dirname, '..'), sources[key]);
  const to = path.relative(process.cwd(), destinations[key]);
  const exists = glob.sync(destinations[key]).length > 0;

  console.log(grey('Writing'), from, green('->'), to, exists ? red('[overwrite]') : green('[create]'));

  fs.writeFileSync(destinations[key], content, 'utf-8');
});


const scaffold = args => {
  // Override arguments, use `--package.name=some-name` to skip prompts
  prompt.override = args;

  getPrompt()
    .then(parseProps(defaults))
    .then(generate(srcContent))
    .then(mergePackages(getPackage(), destinations))
    .then(saveCompiled)
    .then(() => {
      console.log(green('OK'), 'Generation completed', '\n');
      console.log(grey('Run following commands:'));
      console.log('  ', yellow('npm install'));
      console.log('  ', yellow('npm test'));
    })
    .catch(err => console.error(err));
};


// Check if script is run directly
if (require.main === module) {
  scaffold(parse());
}


export default scaffold;
