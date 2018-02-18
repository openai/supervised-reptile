const neuralnet = require('../network');
const {assertClose} = require('./test');

function testAll() {
    testSoftmax();
}

function testSoftmax() {
    const inputs = [-0.683956, 0.552308, 2.014365, 0.301573, 0.380810, -1.711707, 2.367460, -0.768971, -0.382122, -0.762599, 0.726820, 0.864082, -0.407515, -0.673841, 0.914179];
    const outputs = [-2.959975, -1.723711, -0.261655, -0.795709, -0.716472, -2.808988, -0.102008, -3.238439, -2.851590, -2.353419, -0.864000, -0.726738, -1.707645, -1.973971, -0.385951];
    const upstream = [0.512335, 0.802171, -0.659881, 0.955286, 0.625436, -0.741298, -0.290291, 1.084897, -1.432195, -0.166762, -0.636521, 1.235203, -0.097356, -0.188098, 0.385449];
    const inputGrad = [0.478412, 0.685384, -1.163796, 0.576487, 0.215400, -0.791887, 0.285466, 1.109906, -1.395372, -0.207813, -0.818564, 1.026377, -0.115484, -0.201987, 0.317471];

    const inputVar = new neuralnet.Variable(inputs);
    const actualOutput = neuralnet.logSoftmax(inputVar, 3);
    assertClose(outputs, actualOutput.value);
    actualOutput.backward(upstream);
    assertClose(inputVar.gradient, inputGrad);
}

testAll();
console.log('PASS');
