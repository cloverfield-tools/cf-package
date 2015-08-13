const parseProps = (defaults = {}) => props => Object.keys(props)
  .reduce((result, prop) => {
    const [group, key] = prop.split('.');
    if (!result[group]) {
      result[group] = {};
    }
    result[group][key] = props[prop];
    return result;
  }, defaults);

export default parseProps;
