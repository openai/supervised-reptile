(function() {

    // conv2d performs a batched 2-D convolution with 3x3
    // kernels with "same"-style padding.
    //
    // Filters have the following shape:
    // [filter_height, filter_width, in_channels, out_channels].
    function conv2d(input, filters, inRows, inCols, inChannels, stride) {
        var dataSize = inRows * inCols * inChannels;
        var batchSize = input.value.length / dataSize;
        var outChannels = filters.value.length / (3 * 3 * inChannels);
        var filterImages = splitFilters(filters, 3, 3, outChannels);
        var results = [];
        var images = [];

        var rowOff = (inRows - 1) % stride;
        var colOff = (inCols - 1) % stride;

        for (var i = 0; i < batchSize; ++i) {
            var image = new Image(inRows, inCols, inChannels,
                                  input.value.slice(dataSize*i, dataSize*(i+1)));
            for (var j = -1; j + 3 <= inRows + 1; j += stride) {
                for (var k = -1; k + 3 <= inCols + 1; k += stride) {
                    for (var l = 0; l < filterImages.length; ++l) {
                        results.push(convAtSpot(image, filterImages[l], j+rowOff, k+colOff));
                    }
                }
            }
            images.push(image);
        }
        return {
            value: results,
            backward: function(outgrad) {
                var ingrad = [];
                var filterGrads = [];
                for (var i = 0; i < filterImages.length; ++i) {
                    var img = filterImages[i];
                    filterGrads.push(new Image(img.rows, img.cols, img.depth));
                }
                var outgradIdx = 0;
                for (var i = 0; i < batchSize; ++i) {
                    var imageGrad = new Image(inRows, inCols, inChannels);
                    var image = images[i];
                    for (var j = -1; j + 3 <= inRows + 1; j += stride) {
                        for (var k = -1; k + 3 <= inCols + 1; k += stride) {
                            for (var l = 0; l < filterImages.length; ++l) {
                                convGradAtSpot(image, filterImages[l], j+rowOff, k+colOff,
                                    imageGrad, filterGrads[l], outgrad[outgradIdx]);
                                ++outgradIdx;
                            }
                        }
                    }
                    for (var j = 0; j < imageGrad.data.length; ++j) {
                        ingrad.push(imageGrad.data[j]);
                    }
                }
                input.backward(ingrad);
                filters.backward(joinFilters(filterGrads));
            }
        };
    }

    function splitFilters(input, rows, cols, outChannels) {
        var inChannels = input.value.length / (rows * cols * outChannels);
        var filters = [];
        for (var i = 0; i < outChannels; ++i) {
            var filter = [];
            for (var j = i; j < input.value.length; j += outChannels) {
                filter.push(input.value[j]);
            }
            filters.push(new Image(rows, cols, inChannels, filter));
        }
        return filters;
    }

    function joinFilters(filters) {
        var result = [];
        for (var i = 0; i < filters[0].data.length; ++i) {
            for (var j = 0; j < filters.length; ++j) {
                result.push(filters[j].data[i]);
            }
        }
        return result;
    }

    function convAtSpot(image, filter, row, col) {
        var sum = 0;
        for (var i = 0; i < filter.rows; ++i) {
            for (var j = 0; j < filter.cols; ++j) {
                for (var k = 0; k < filter.depth; ++k) {
                    sum += image.get(row+i, col+j, k) * filter.get(i, j, k);
                }
            }
        }
        return sum;
    }

    function convGradAtSpot(image, filter, row, col, imageGrad, filterGrad, scale) {
        for (var i = 0; i < filter.rows; ++i) {
            for (var j = 0; j < filter.cols; ++j) {
                for (var k = 0; k < filter.depth; ++k) {
                    imageGrad.add(row+i, col+j, k, filter.get(i, j, k) * scale);
                    filterGrad.add(i, j, k, image.get(row+i, col+j, k) * scale);
                }
            }
        }
    }

    function Image(rows, cols, depth, data) {
        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        if (data) {
            this.data = data;
        } else {
            this.data = [];
            for (var i = 0; i < rows * cols * depth; ++i) {
                this.data.push(0);
            }
        }
        if (this.data.length !== rows * cols * depth) {
            throw Error('invalid data size');
        }
    }

    Image.prototype.get = function(row, col, channel) {
        if (this.outOfBounds(row, col)) {
            return 0;
        }
        return this.data[this.index(row, col, channel)];
    };

    Image.prototype.set = function(row, col, channel, val) {
        if (!this.outOfBounds(row, col)) {
            this.data[this.index(row, col, channel)] = val;
        }
    };

    Image.prototype.add = function(row, col, channel, val) {
        if (!this.outOfBounds(row, col)) {
            this.data[this.index(row, col, channel)] += val;
        }
    };

    Image.prototype.index = function(row, col, channel) {
        return ((row * this.cols) + col) * this.depth + channel;
    };

    Image.prototype.outOfBounds = function(row, col) {
        return row < 0 || row >= this.rows || col < 0 || col >= this.cols;
    };

    if ('undefined' !== typeof window) {
        window.neuralnet = (window.neuralnet || {});
        window.neuralnet.conv2d = conv2d;
    } else {
        module.exports = conv2d;
    }

})();
