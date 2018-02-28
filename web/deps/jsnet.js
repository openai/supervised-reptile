(function() {
function Tensor(shape, data) {
    var dataSize = shapeProduct(shape);
    this.shape = shape;
    if ('undefined' !== typeof Float32Array) {
        if (!data) {
            this.data = new Float32Array(dataSize);
        } else {
            if (data.length !== dataSize) {
                throw Error('invalid data size');
            }
            this.data = new Float32Array(data);
        }
    } else {
        if (!data) {
            this.data = [];
            for (var i = 0; i < dataSize; ++i) {
                this.data.push(0);
            }
        } else {
            if (data.length !== dataSize) {
                throw Error('invalid data size');
            }
            this.data = [];
            for (var i = 0; i < dataSize; ++i) {
                this.data.push(data[i]);
            }
        }
    }
}

Tensor.prototype.copy = function() {
    return new Tensor(this.shape, this.data);
};

Tensor.prototype.scale = function(scale) {
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] *= scale;
    }
    return this;
};

Tensor.prototype.addScalar = function(scalar) {
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] += scalar;
    }
    return this;
}

Tensor.prototype.add = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] += other.data[i];
    }
    return this;
};

Tensor.prototype.sub = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] -= other.data[i];
    }
    return this;
};

Tensor.prototype.mul = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] *= other.data[i];
    }
    return this;
};

Tensor.prototype.div = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] /= other.data[i];
    }
    return this;
};

Tensor.prototype.reshape = function(shape) {
    if (shapeProduct(shape) !== shapeProduct(this.shape)) {
        throw Error('cannot reshape from [' + this.shape + '] to [' + shape + ']');
    }
    this.shape = shape;
    return this;
};

Tensor.prototype.repeated = function(repeats) {
    var result = new Tensor([repeats].concat(this.shape));
    for (var i = 0; i < repeats; ++i) {
        var startIdx = i * this.data.length;
        for (var j = 0; j < this.data.length; ++j) {
            result.data[startIdx + j] = this.data[j];
        }
    }
    return result;
};

Tensor.prototype.sumOuter = function() {
    if (this.shape.length === 0) {
        return this.copy();
    }
    var result = new Tensor(this.shape.slice(1));
    var chunkSize = this.data.length / this.shape[0];
    for (var i = 0; i < this.shape[0]; ++i) {
        var offset = chunkSize * i;
        for (var j = 0; j < chunkSize; ++j) {
            result.data[j] += this.data[j + offset];
        }
    }
    return result;
};

Tensor.prototype._assertSameShape = function(other) {
    if (this.shape.length !== other.shape.length) {
        throw Error('shapes not equal');
    }
    for (var i = 0; i < this.shape.length; ++i) {
        if (this.shape[i] !== other.shape[i]) {
            throw Error('shapes not equal');
        }
    }
};

function shapeProduct(shape) {
    var dataSize = 1;
    for (var i = 0; i < shape.length; ++i) {
        dataSize *= shape[i];
    }
    return dataSize;
}
function Variable(value) {
    this.value = value;
    this.gradient = new Tensor(this.value.shape);
}

Variable.prototype.backward = function(outGrad) {
    this.gradient.add(outGrad);
};

Variable.prototype.clearGrad = function() {
    this.gradient = new Tensor(this.gradient.shape);
};

function pool(input, f) {
    var poolVar = new Variable(input.value);
    var result = f(poolVar);
    return {
        value: result.value,
        backward: function(outGrad) {
            poolVar.clearGrad();
            result.backward(outGrad);
            input.backward(poolVar.gradient);
        }
    };
}

function scale(input, scale) {
    return {
        value: input.value.copy().scale(scale),
        backward: function(outGrad) {
            input.backward(outGrad.copy().scale(scale));
        }
    };
}

function addScalar(input, scalar) {
    return {
        value: input.value.copy().addScalar(scalar),
        backward: function(outGrad) {
            input.backward(outGrad);
        }
    };
}

function add(input1, input2) {
    return {
        value: input1.value.copy().add(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad);
            input2.backward(outGrad);
        }
    };
}

function sub(input1, input2) {
    return {
        value: input1.value.copy().sub(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad);
            input2.backward(outGrad.copy().scale(-1));
        }
    };
}

function mul(input1, input2) {
    return {
        value: input1.value.copy().mul(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad.copy().mul(input2.value));
            input2.backward(outGrad.copy().mul(input1.value));
        }
    };
}

function div(input1, input2) {
    return {
        value: input1.value.copy().div(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad.copy().div(input2.value));
            var denomGrad = outGrad.copy().mul(input1.value);
            denomGrad.div(input2.value).div(input2.value).scale(-1);
            input2.backward(denomGrad);
        }
    };
}

function reshape(input, shape) {
    return {
        value: input.value.copy().reshape(shape),
        backward: function(outGrad) {
            input.backward(outGrad.copy().reshape(input.value.shape));
        }
    };
}

function repeated(input, repeats) {
    return {
        value: input.value.repeated(repeats),
        backward: function(outGrad) {
            input.backward(outGrad.sumOuter());
        }
    };
}

function sumOuter(input) {
    if (input.value.shape.length === 0) {
        return input;
    }
    return {
        value: input.value.sumOuter(),
        backward: function(outGrad) {
            input.backward(outGrad.repeated(input.value.shape[0]));
        }
    };
}

function broadcast(input, shape) {
    if (!canBroadcast(input.value.shape, shape)) {
        throw Error('cannot broadcast from shape [' + input.value.shape + '] to [' + shape + ']');
    }
    while (input.value.shape.length < shape.length) {
        input = repeated(input, shape[shape.length - input.value.shape.length - 1]);
    }
    return input;
}

function canBroadcast(src, dst) {
    if (src.length > dst.length) {
        return false;
    }
    for (var i = 0; i < src.length; ++i) {
        if (src[i] !== dst[dst.length - src.length + i]) {
            return false;
        }
    }
    return true;
}
function padImages(images, top, right, bottom, left) {
    if (images.value.shape.length !== 4) {
        throw Error('expected 4-D image tensor');
    }
    var padded = new Tensor([
        images.value.shape[0],
        images.value.shape[1] + top + bottom,
        images.value.shape[2] + left + right,
        images.value.shape[3]
    ]);
    for (var i = 0; i < images.value.shape[0]; ++i) {
        var srcIdx = i * shapeProduct(images.value.shape.slice(1));
        var dstIdx = i * shapeProduct(padded.shape.slice(1));
        for (var j = 0; j < images.value.shape[1]; ++j) {
            var jRowSize = shapeProduct(images.value.shape.slice(2));
            var srcStart = srcIdx + j * jRowSize;
            var dstStart = (dstIdx + (j + top) * shapeProduct(padded.shape.slice(2)) +
                left * padded.shape[3]);
            for (var k = 0; k < jRowSize; ++k) {
                padded.data[dstStart + k] = images.value.data[srcStart + k];
            }
        }
    }
    return {
        value: padded,
        backward: function(outGrad) {
            var unpadded = new Tensor(images.value.shape);
            for (var i = 0; i < unpadded.shape[0]; ++i) {
                var srcIdx = i * shapeProduct(outGrad.shape.slice(1));
                var dstIdx = i * shapeProduct(unpadded.shape.slice(1));
                for (var j = 0; j < unpadded.shape[1]; ++j) {
                    var jRowSize = shapeProduct(unpadded.shape.slice(2));
                    var dstStart = dstIdx + j * jRowSize;
                    var srcStart = (srcIdx + (j + top) * shapeProduct(outGrad.shape.slice(2)) +
                        left * outGrad.shape[3]);
                    for (var k = 0; k < jRowSize; ++k) {
                        unpadded.data[dstStart + k] = outGrad.data[srcStart + k];
                    }
                }
            }
            images.backward(unpadded);
        }
    };
}

function imagePatches(images, windowHeight, windowWidth, strideY, strideX) {
    if (images.value.shape.length !== 4) {
        throw Error('expected 4-D image tensor');
    } else if (images.value.shape[1] < windowHeight || images.value.shape[2] < windowWidth) {
        throw Error('window larger than image');
    }
    var rowSize = shapeProduct(images.value.shape.slice(2));
    var depth = images.value.shape[3];
    var result = new Tensor([
        images.value.shape[0],
        1 + Math.floor((images.value.shape[1] - windowHeight) / strideY),
        1 + Math.floor((images.value.shape[2] - windowWidth) / strideX),
        windowHeight,
        windowWidth,
        depth
    ]);
    var dstIdx = 0;
    for (var i = 0; i < result.shape[0]; ++i) {
        var batchStart = i * shapeProduct(images.value.shape.slice(1));
        for (var j = 0; j < result.shape[1]; ++j) {
            for (var k = 0; k < result.shape[2]; ++k) {
                var srcIdx = batchStart + (j * strideY * rowSize) + (k * strideX * depth);
                for (var l = 0; l < windowHeight; ++l) {
                    var rowStart = srcIdx + l * rowSize;
                    for (var m = 0; m < windowWidth * depth; ++m) {
                        result.data[dstIdx++] = images.value.data[rowStart + m];
                    }
                }
            }
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = new Tensor(images.value.shape);
            var outIdx = 0;
            for (var i = 0; i < result.shape[0]; ++i) {
                var batchStart = i * shapeProduct(images.value.shape.slice(1));
                for (var j = 0; j < result.shape[1]; ++j) {
                    for (var k = 0; k < result.shape[2]; ++k) {
                        var srcIdx = batchStart + (j * strideY * rowSize) + (k * strideX * depth);
                        for (var l = 0; l < windowHeight; ++l) {
                            var rowStart = srcIdx + l * rowSize;
                            for (var m = 0; m < windowWidth * depth; ++m) {
                                inGrad.data[rowStart + m] += outGrad.data[outIdx++];
                            }
                        }
                    }
                }
            }
            images.backward(inGrad);
        }
    };
}
function matmul(mat1, mat2) {
    if (mat1.value.shape.length !== 2 || mat2.value.shape.length !== 2) {
        throw Error('matrices must be two-dimensional');
    } else if (mat1.value.shape[1] !== mat2.value.shape[0]) {
        throw Error('inner dimension mismatch');
    }
    var resultData = new Tensor([mat1.value.shape[0], mat2.value.shape[1]]);
    var idx = 0;
    for (var i = 0, rows = mat1.value.shape[0]; i < rows; ++i) {
        var rowOffset = mat1.value.shape[1] * i;
        for (var j = 0, cols = mat2.value.shape[1]; j < cols; ++j) {
            var sum = 0;
            for (var k = 0, inner = mat1.value.shape[1]; k < inner; ++k) {
                sum += mat1.value.data[rowOffset + k] * mat2.value.data[k*cols + j];
            }
            resultData.data[idx++] = sum;
        }
    }
    return {
        value: resultData,
        backward: function(outGrad) {
            var mat1Grad = new Tensor(mat1.value.shape);
            var mat2Grad = new Tensor(mat2.value.shape);
            var idx = 0;
            for (var i = 0, rows = mat1.value.shape[0]; i < rows; ++i) {
                var rowOffset = mat1.value.shape[1] * i;
                for (var j = 0, cols = mat2.value.shape[1]; j < cols; ++j) {
                    var elemGrad = outGrad.data[idx++];
                    for (var k = 0, inner = mat1.value.shape[1]; k < inner; ++k) {
                        mat1Grad.data[rowOffset + k] += elemGrad * mat2.value.data[k*cols + j];
                        mat2Grad.data[k*cols + j] += elemGrad * mat1.value.data[rowOffset + k];
                    }
                }
            }
            mat1.backward(mat1Grad);
            mat2.backward(mat2Grad);
        }
    };
}

function conv2d(images, filters, strideY, strideX) {
    if (filters.value.shape.length !== 4) {
        throw new Error('expected 4-D filter tensor');
    }
    strideY = (strideY || 1);
    strideX = (strideX || strideY);
    var patches = imagePatches(images, filters.value.shape[0], filters.value.shape[1],
        strideY, strideX);
    var leftMatrix = reshape(patches, [shapeProduct(patches.value.shape.slice(0, 3)),
        shapeProduct(patches.value.shape.slice(3))]);
    var rightMatrix = reshape(filters, [shapeProduct(filters.value.shape.slice(0, 3)),
        filters.value.shape[3]]);
    var product = matmul(leftMatrix, rightMatrix);
    return reshape(product, patches.value.shape.slice(0, 3).concat([filters.value.shape[3]]));
}
function relu(input) {
    var result = input.value.copy();
    for (var i = 0; i < result.data.length; ++i) {
        if (result.data[i] < 0) {
            result.data[i] = 0;
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < outGrad.data.length; ++i) {
                if (input.value.data[i] < 0) {
                    inGrad.data[i] = 0;
                }
            }
            input.backward(inGrad);
        }
    };
}

function rsqrt(input) {
    return pow(input, -0.5);
}

function square(input) {
    return pow(input, 2);
}

function pow(input, power) {
    var result = input.value.copy();
    for (var i = 0; i < result.data.length; ++i) {
        result.data[i] = Math.pow(result.data[i], power);
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < inGrad.data.length; ++i) {
                inGrad.data[i] *= power * Math.pow(input.value.data[i], power - 1);
            }
            input.backward(inGrad);
        }
    }
}
var DEFAULT_EPSILON = 0.001;

function normalizeChannels(input, epsilon) {
    epsilon = epsilon || DEFAULT_EPSILON;
    return pool(input, function(input) {
        var centered = sub(input, broadcast(channelMean(input), input.value.shape));
        var variance = addScalar(channelMean(square(centered)), epsilon);
        return mul(centered, broadcast(rsqrt(variance), input.value.shape));
    });
}

function channelMean(input) {
    var count = 1;
    while (input.value.shape.length > 1) {
        count *= input.value.shape[0];
        input = sumOuter(input);
    }
    return scale(input, 1 / count);
}

function logSoftmax(input) {
    var batchSize = shapeProduct(input.value.shape.slice(0, input.value.shape.length - 1));
    var channels = input.value.shape[input.value.shape.length - 1];
    var logSumExps = [];
    var result = input.value.copy();
    for (var i = 0; i < batchSize; ++i) {
        var max = -Infinity;
        var start = channels * i;
        for (var j = 0; j < channels; ++j) {
            max = Math.max(max, input.value.data[start + j]);
        }
        var logSumExp = 0;
        for (var j = 0; j < channels; ++j) {
            logSumExp += Math.exp(input.value.data[start + j] - max);
        }
        logSumExp = Math.log(logSumExp) + max;
        logSumExps[i] = logSumExp;
        for (var j = 0; j < channels; ++j) {
            result.data[start + j] -= logSumExp;
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < batchSize; ++i) {
                var start = channels * i;
                var gradSum = 0;
                for (var j = 0; j < channels; ++j) {
                    gradSum += outGrad.data[start + j];
                }
                for (var j = 0; j < channels; ++j) {
                    var normalizeGrad = Math.exp(input.value.data[start + j] - logSumExps[i]);
                    inGrad.data[start + j] = outGrad.data[start + j] - normalizeGrad * gradSum;
                }
            }
            input.backward(inGrad);
        }
    };
}
var exportObj = {
    Tensor: Tensor,
    Variable: Variable,
    pool: pool,
    scale: scale,
    addScalar: addScalar,
    add: add,
    sub: sub,
    mul: mul,
    div: div,
    reshape: reshape,
    repeated: repeated,
    sumOuter: sumOuter,
    broadcast: broadcast,
    canBroadcast: canBroadcast,
    matmul: matmul,
    conv2d: conv2d,
    padImages: padImages,
    imagePatches: imagePatches,
    relu: relu,
    rsqrt: rsqrt,
    square: square,
    pow: pow,
    normalizeChannels: normalizeChannels,
    channelMean: channelMean,
    logSoftmax: logSoftmax
};

if ('undefined' !== typeof window) {
    window.jsnet = exportObj;
} else if ('undefined' !== typeof self) {
    self.jsnet = exportObj;
} else {
    module.exports = exportObj;
}
})();
