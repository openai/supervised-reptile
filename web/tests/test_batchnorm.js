const neuralnet = require('../network');
const {assertClose} = require('./test');

function testAll() {
    testBatchNorm();
}

function testBatchNorm() {
    const inputs = [0.610561, -1.471499, -0.473854, 1.874188, -0.633909, 0.428393, 1.094642, 1.396149, -0.760894, 0.609285, 0.474102, 0.665654, 1.020200, -0.785205, -1.297708];
    const outputs = [-1.440600, -2.450581, 0.384874, 1.285667, -1.625637, 1.611461, -0.396197, 0.373770, -0.005351, -1.443353, -0.534355, 1.934013, -0.556807, -1.774648, -0.735139];
    const upstream = [0.883774, -0.207172, 0.670983, 1.023162, -0.627646, 0.703418, 0.956206, 0.533413, 0.516775, -0.363842, 0.421499, 1.161440, 0.167439, -0.955223, 0.354755];
    const inputGrad = [1.340573, 0.499762, 0.068362, -0.071603, -0.270766, -0.287965, 0.840649, 0.008954, -0.013892, -1.349419, 0.291069, 0.229409, -0.760199, -0.529020, 0.004086];
    const betaGrad = [2.666739, -0.835129, 3.407370];
    const beta = [-0.510258, -1.202290, 0.637972];

    const inputVar = new neuralnet.Variable(inputs);
    const betaVar = new neuralnet.Variable(beta);
    const actualOutput = neuralnet.batchNorm(inputVar, betaVar, 3);
    assertClose(outputs, actualOutput.value);
    actualOutput.backward(upstream);
    assertClose(inputVar.gradient, inputGrad);
    assertClose(betaVar.gradient, betaGrad);
}

testAll();
console.log('PASS');
