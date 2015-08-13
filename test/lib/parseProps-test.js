import {test} from 'blue-tape';
import parseProps from '../../lib/parseProps';


test('Parse props', t => {
  t.ok(parseProps instanceof Function, 'should be function');

  t.ok(parseProps() instanceof Function, 'should return function');

  t.deepEqual(parseProps()({'x.y': 2}), {x: {y: 2}},
    'should parse dot-delimited props into nested objects');

  t.deepEqual(parseProps({x: {a: 1}})({'x.y': 2}), {x: {a: 1, y: 2}},
    'should merge parsed data into defaults object');

  t.end();
});
