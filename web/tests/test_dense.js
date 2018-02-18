const neuralnet = require('../network');
const {assertClose} = require('./test');

function testAll() {
    testDense();
}

function testDense() {
    const inputs = [1.100552, -0.314686, -0.673426, 0.418534, 1.412955, -0.039111, -1.803552, -1.755020, -0.172488, -0.151655, 2.029588, -0.922730, -0.140666, 2.117852, 1.052755];
    const outputs = [-0.636578, 0.203870, -1.121999, -1.208630, 0.442963, -1.053417, 0.652273, -0.645311, 0.810503, 2.049156, 0.000151, 1.849319, 2.038154, -0.532460, 1.226921, -0.626199, 0.458899, -2.057731, 1.775115, 0.189126];
    const upstream = [-0.332166, -0.056889, -0.086666, -0.399623, 0.040030, 0.107732, 0.069870, 0.575836, -0.612009, 0.020008, -0.237255, -1.117946, 0.049132, -0.065069, 0.564743, -1.352844, 0.474517, 1.034981, -0.689375, -0.532059];
    const inputGrad = [0.744725, -0.136425, 0.122436, -0.644557, -0.103425, 0.150075, 1.707872, -0.312978, -0.048947, 0.806570, 0.691769, -0.426065, 0.037215, -0.752197, -1.555941];
    const weightsGrad = [0.680778, -0.189324, 0.373089, 2.097483, 2.339849, 2.194881, 0.228583, -0.971134, 0.781902, 1.180268, -1.150294, 1.127610];
    const weights = [-0.918869, -0.449354, -0.674544, -0.889549, 0.562998, -0.632958, 0.668404, -0.181430, -0.819469, -0.741320, 0.251387, 0.425776];

    const inputVar = new neuralnet.Variable(inputs);
    const weightsVar = new neuralnet.Variable(weights);
    const actualOutput = neuralnet.dense(inputVar, weightsVar, 3, 4);
    assertClose(outputs, actualOutput.value);
    actualOutput.backward(upstream);
    assertClose(inputVar.gradient, inputGrad);
    assertClose(weightsVar.gradient, weightsGrad);
}

testAll();
console.log('PASS');
