module.exports = function(name, f) {
    const start = new Date().getTime();
    let count = 0;
    while (new Date().getTime() - start < 1000) {
        f();
        ++count;
    }
    console.log(name + ' took ' + ((new Date().getTime() - start) / count) + ' ms');
};
