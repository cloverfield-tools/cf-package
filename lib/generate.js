import {compile} from 'handlebars';

const generate = files => props =>
  files
    .map(file => compile(file))
    .map(compiler => compiler(props));

export default generate;
