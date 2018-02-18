const neuralnet = require('../network');
const conv2d = require('../conv');
const bench = require('./bench');

function benchmarkOmniglot() {
    const batch = 5
    const kernels = [1, 64, 64, 64].map((s) => new neuralnet.Variable(neuralnet.zeros(s*64*3*3)));
    const input = new neuralnet.Variable(neuralnet.zeros(batch * 28 * 28));
    bench('omniglot', () => {
        output = conv2d(input, kernels[0], 28, 28, 1, 2);
        output = conv2d(output, kernels[1], 14, 14, 64, 2);
        output = conv2d(output, kernels[2], 7, 7, 64, 2);
        output = conv2d(output, kernels[3], 4, 4, 64, 2);
        output.backward(neuralnet.zeros(batch * 2 * 2 * 64));
    });
}

benchmarkOmniglot();
