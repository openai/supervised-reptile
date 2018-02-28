var ADAM_LR = 0.001;
var NUM_STEPS = 5;

self.onmessage = function(msg) {
    var classes = msg.data.classes;
    var data = msg.data.data;

    var tensorSize = data.length / (classes + 1);
    var size = Math.sqrt(tensorSize);
    var trainData = data.slice(0, tensorSize * classes);

    var parameters = trainedInit();
    var images = new jsnet.Tensor([classes, size, size, 1], trainData);
    var losses = trainNetwork(parameters, images, NUM_STEPS);

    images = new jsnet.Tensor([classes+1, size, size, 1], data);
    var output = applyNetwork(parameters, images).value;

    var probs = [];
    for (var i = 0; i < classes; ++i) {
        probs.push(Math.exp(output.data[i + classes * classes]));
    }
    self.postMessage({losses: losses, probs: probs});
};

function trainNetwork(parameters, images, steps) {
    var losses = [];
    for (var i = 0; i < steps; ++i) {
        var outputs = applyNetwork(parameters, images);
        var loss = computeLoss(outputs);
        losses.push(loss.value.data[0]);
        loss.backward(new jsnet.Tensor([], [1]));
        for (var j = 0; j < parameters.length; ++j) {
            var param = parameters[j];
            param.gradient.mul(param.adamRate);
            param.gradient.scale(ADAM_LR);
            param.value.add(param.gradient);
            param.clearGrad();
        }
    }
    return losses;
}

function computeLoss(outputs) {
    var ways = outputs.value.shape[0];
    var rawMask = [];
    for (var i = 0; i < ways; ++i) {
        for (var j = 0; j < ways; ++j) {
            if (i == j) {
                rawMask.push(1);
            } else {
                rawMask.push(0);
            }
        }
    }
    var mask = new jsnet.Tensor([ways, ways], rawMask);
    var masked = jsnet.mul(outputs, new jsnet.Variable(mask));
    return jsnet.sumOuter(jsnet.sumOuter(masked));
}

function applyNetwork(parameters, images) {
    var output = new jsnet.Variable(images);
    for (var i = 0; i < 4; ++i) {
        output = applyConv(output, parameters.slice(i*3, (i+1)*3));
    }
    output = jsnet.reshape(output, [
        output.value.shape[0],
        output.value.shape[1] * output.value.shape[2] * output.value.shape[3]
    ]);
    output = applyDense(output, parameters.slice(12, 14));
    return jsnet.logSoftmax(output);
}

function applyConv(inputs, parameters, i) {
    var kernel = parameters[0];
    var gamma = parameters[1];
    var beta = parameters[2];
    var output = inputs;
    if (output.value.shape[1] % 2 === 1) {
        output = jsnet.padImages(inputs, 1, 1, 1, 1);
    } else {
        output = jsnet.padImages(inputs, 0, 1, 1, 0);
    }
    output = jsnet.conv2d(output, kernel, 2, 2);
    output = jsnet.normalizeChannels(output);
    output = jsnet.mul(output, jsnet.broadcast(gamma, output.value.shape));
    output = jsnet.add(output, jsnet.broadcast(beta, output.value.shape));
    return jsnet.relu(output);
}

function applyDense(inputs, parameters) {
    var kernel = parameters[0];
    var bias = parameters[1];
    var output = jsnet.matmul(inputs, kernel);
    return jsnet.add(output, jsnet.broadcast(bias, output.value.shape));
}

function randomInit(numClasses) {
    var parameters = [];
    for (var i = 0; i < 4; ++i) {
        var inChans = [1, 24, 24, 24][i];
        var filters = [];
        var gamma = [];
        var beta = [];
        for (var j = 0; j < inChans * 3 * 3 * 24; ++j) {
            filters.push((Math.random() - 0.5) / 50);
        }
        for (var j = 0; j < 24; ++j) {
            gamma.push(1);
            beta.push(0);
        }
        parameters.push(new jsnet.Variable(new jsnet.Tensor([3, 3, inChans, 24], filters)));
        parameters.push(new jsnet.Variable(new jsnet.Tensor([24], gamma)));
        parameters.push(new jsnet.Variable(new jsnet.Tensor([24], beta)));
    }
    var weightMatrix = [];
    for (var i = 0; i < 96 * numClasses; ++i) {
        weightMatrix.push((Math.random() - 0.5) / 10);
    }
    parameters.push(new jsnet.Variable(new jsnet.Tensor([96, numClasses], weightMatrix)));
    var biases = [];
    for (var i = 0; i < numClasses; ++i) {
        biases.push(0);
    }
    parameters.push(new jsnet.Variable(new jsnet.Tensor([numClasses], biases)));

    var adamScales = [
        3.238116836475661,
        5.385047353671038,
        5.637935420477302,
        70.68438036761029,
        5.614735134965089,
        5.880044363499613,
        52.143705309651224,
        3.306547626966237,
        5.350009339356334,
        1.8687280293343194,
        2.591797130259263,
        2.9093116107992216,
        8.542032513733227,
        2.2572890209671423
    ];

    for (var i = 0; i < parameters.length; ++i) {
        var param = parameters[i];
        var scales = [];
        for (var j = 0; j < param.value.data.length; ++j) {
            scales.push(adamScales[i]);
        }
        parameters[i].adamRate = new jsnet.Tensor(param.value.shape, scales);
    }

    return parameters;
}

function trainedInit() {
    var result = [];

    for (var i = 0; i < trainedParameters.length; i += 2) {
        var param = new jsnet.Variable(trainedParameters[i].copy());
        param.adamRate = trainedParameters[i + 1].copy();
        for (var j = 0; j < param.adamRate.data.length; ++j) {
            param.adamRate.data[j] = 1 / (Math.sqrt(param.adamRate.data[j]) + 1e-5);
        }
        result.push(param);
    }

    return result;
}
