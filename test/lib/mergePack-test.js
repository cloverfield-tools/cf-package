import {test} from 'blue-tape';
import {mergePackages} from '../../lib/mergePack';
import path from 'path';

test('Merge Packages', t => {

  const packPath = path.join(process.cwd(), 'package.json');
  const templatedFile = JSON.stringify({
    name: 'templatePackage',
    description: 'This should overwrite stuff'
  });
  const originalFile = JSON.stringify({
    name: 'originalPackage',
    description: 'This should be overwritten by a template',
    scripts: {
      start: 'do a lotta stuff'
    }
  });
  const mergedFile = JSON.stringify({
    name: 'templatePackage',
    description: 'This should overwrite stuff',
    scripts: {
      start: 'do a lotta stuff'
    }
  });

  t.ok(mergePackages instanceof Function, 'should be function');

  t.ok(mergePackages() instanceof Function, 'should return function');

  t.deepEqual(mergePackages(false, [packPath])([templatedFile]), [templatedFile],
    'should return the generated package when there is no pre-existing package.json');

  t.deepEqual(mergePackages(originalFile, [packPath])([templatedFile]), [mergedFile],
    'should merge an existing package.json with template and return results');

  t.end();
});
