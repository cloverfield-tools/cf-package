import {test} from 'blue-tape';
import {mergePackages} from '../../lib/mergePack';
import path from 'path';

test('Merge Packages', t => {

  const packPath = path.join(process.cwd(), 'package.json');
  const templatedFile = JSON.stringify({
    name: 'templatePackage',
    description: 'This should not overwrite stuff',
    main: 'src/index.js',
    scripts: {
      start: 'do a lotta new stuff',
      test: 'mocha this package'
    }
  });
  const originalFile = {
    name: 'originalPackage',
    description: 'This should not be overwritten by a template',
    scripts: {
      start: 'do a lotta old stuff',
      lint: 'lint this package'
    }
  };
  const mergedFile = {
    name: 'originalPackage',
    description: 'This should not be overwritten by a template',
    main: 'src/index.js',
    scripts: {
      start: 'do a lotta old stuff',
      lint: 'lint this package',
      test: 'mocha this package'
    }
  };

  t.ok(mergePackages instanceof Function, 'should be function');

  t.ok(mergePackages() instanceof Function, 'should return function');

  t.deepEqual(mergePackages(false, [packPath])([templatedFile]), [templatedFile],
    'should return the generated package when there is no pre-existing package.json');

  const merger = mergePackages(originalFile, [packPath])([templatedFile])[0];

  t.deepEqual(JSON.parse(merger), mergedFile,
    'should merge an existing package.json with template and return results');

  t.ok(merger.match(/\s\s/), 'merged package should be formatted with spaces');

  t.end();
});
