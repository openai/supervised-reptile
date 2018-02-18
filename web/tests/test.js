const ERROR_DELTA = 1e-3;

function assertClose(vec1, vec2) {
    if (vec1.length !== vec2.length) {
        throw Error('vectors not equal in length');
    }
    for (let i = 0; i < vec1.length; ++i) {
        if (isNaN(vec1[i]) || isNaN(vec2[i])) {
            throw Error('NaN in vector');
        }
        if (Math.abs(vec1[i] - vec2[i]) > ERROR_DELTA) {
            throw Error('vectors not equal');
        }
    }
}

module.exports = {assertClose: assertClose};
