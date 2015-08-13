import {test} from 'blue-tape';
import generate from '../../lib/generate';


test('Generate', t => {
  t.ok(generate instanceof Function, 'should be function');

  t.ok(generate() instanceof Function, 'should return function');

  t.deepEqual(generate(['test {{x.y}}'])({x: {y: 1}}), ['test 1'],
    'should fill template with values');

  t.deepEqual(generate(['test {{x.y}}', 'another {{x.y}}'])({x: {y: 1}}), ['test 1', 'another 1'],
    'should fill multiple templates with values');

  t.end();
});
