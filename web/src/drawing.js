(function() {

    var LABEL_SIZE = 20;
    var UPSAMPLE = 2;
    var LINE_WIDTH = 0.05;
    var CENTERED_PAD = 0.9;

    function DrawingCell(size) {
        this.onChange = function() {};

        this.size = size;

        this.element = document.createElement('div');
        this.element.className = 'few-shot-cell';

        this._canvas = document.createElement('canvas');
        this._canvas.className = 'few-shot-cell-canvas';
        this._canvas.width = size * UPSAMPLE;
        this._canvas.height = size * UPSAMPLE;
        this.element.appendChild(this._canvas);

        this._emptyLabel = document.createElement('label');
        this._emptyLabel.className = 'few-shot-cell-empty-label';
        this._emptyLabel.textContent = 'Draw Here';
        this.element.appendChild(this._emptyLabel);

        this._paths = [];
        this._redraw();

        this._enabled = false;
    }

    DrawingCell.prototype.enableInteraction = function() {
        if (!this._enabled) {
            this._enabled = true;
            this._registerMouse();
            if ('ontouchstart' in document.documentElement) {
                this._registerTouch();
            }
        }
    };

    DrawingCell.prototype.setPaths = function(paths) {
        this._paths = paths;
        this._redraw();
    };

    DrawingCell.prototype.empty = function() {
        return this._paths.length === 0;
    };

    DrawingCell.prototype.clear = function() {
        this.setPaths([]);
    };

    DrawingCell.prototype.tensor = function(size) {
        var canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;

        var ctx = canvas.getContext('2d');
        ctx.strokeStyle = 'black';
        ctx.lineWidth = LINE_WIDTH * size;
        ctx.lineCap = 'square';
        ctx.lineJoin = 'square';
        drawPaths(ctx, size, this._centeredPaths());

        var data = ctx.getImageData(0, 0, size, size).data;
        var grayscale = [];
        for (var i = 3; i < data.length; i += 4) {
            if (data[i] > 128) {
                grayscale.push(0);
            } else {
                grayscale.push(1);
            }
        }
        return new jsnet.Tensor([size, size, 1], grayscale);
    };

    DrawingCell.prototype._redraw = function() {
        if (this._paths.length === 0) {
            this._emptyLabel.style.display = 'block';
        } else {
            this._emptyLabel.style.display = 'none';
        }
        var ctx = this._canvas.getContext('2d');
        var size = this.size * UPSAMPLE;
        ctx.clearRect(0, 0, size, size);

        ctx.strokeStyle = '#777';
        ctx.lineWidth = LINE_WIDTH * size;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        drawPaths(ctx, size, this._paths);
    };

    DrawingCell.prototype._registerMouse = function() {
        this._canvas.addEventListener('mousedown', function(e) {
            var mouseUp = function() {
                this.onChange();
                removeEvents();
            }.bind(this);
            var mouseMove = function(e) {
                this._paths[this._paths.length - 1].push(this._mousePosition(e));
                this._redraw();
            }.bind(this);
            var removeEvents = function() {
                window.removeEventListener('mouseup', mouseUp);
                window.removeEventListener('mousemove', mouseMove);
            };
            window.addEventListener('mouseup', mouseUp);
            window.addEventListener('mouseleave', mouseUp);
            window.addEventListener('mousemove', mouseMove);
            this._paths.push([this._mousePosition(e)]);
            this._redraw();
        }.bind(this));
    };

    DrawingCell.prototype._registerTouch = function() {
        this._canvas.addEventListener('touchstart', function(e) {
            e.preventDefault();
            this._paths.push([this._touchPosition(e)]);
        }.bind(this));
        this._canvas.addEventListener('touchmove', function(e) {
            e.preventDefault();
            this._paths[this._paths.length - 1].push(this._touchPosition(e));
            this._redraw();
        }.bind(this));
        var onEnd = function() {
            this.onChange();
        }.bind(this);
        this._canvas.addEventListener('touchend', onEnd);
        this._canvas.addEventListener('touchcancel', onEnd);
    };

    DrawingCell.prototype._mousePosition = function(e) {
        var rect = this._canvas.getBoundingClientRect();
        return [(e.clientX - rect.left) / this._canvas.offsetWidth,
            (e.clientY - rect.top) / this._canvas.offsetHeight];
    };

    DrawingCell.prototype._touchPosition = function(e) {
        var rect = this._canvas.getBoundingClientRect();
        var touch = e.changedTouches[0];
        return [(touch.clientX - rect.left) / this._canvas.offsetWidth,
            (touch.clientY - rect.top) / this._canvas.offsetHeight];
    };

    DrawingCell.prototype._centeredPaths = function() {
        var minX = Infinity, maxX = -Infinity;
        var minY = Infinity, maxY = -Infinity;
        for (var i = 0; i < this._paths.length; ++i) {
            var path = this._paths[i];
            for (var j = 0; j < path.length; ++j) {
                var point = path[j];
                minX = Math.min(minX, point[0]);
                maxX = Math.max(maxX, point[0]);
                minY = Math.min(minY, point[1]);
                maxY = Math.max(maxY, point[1]);
            }
        }
        var sideLength = Math.max(maxX-minX, maxY-minY) * (1 + CENTERED_PAD);
        var topX = (maxX+minX)/2 - sideLength/2;
        var topY = (maxY+minY)/2 - sideLength/2;
        var result = [];
        for (var i = 0; i < this._paths.length; ++i) {
            var path = this._paths[i];
            var newPath = [];
            for (var j = 0; j < path.length; ++j) {
                var point = path[j];
                newPath.push([(point[0] - topX) / sideLength, (point[1] - topY) / sideLength]);
            }
            result.push(newPath);
        }
        return result;
    };

    function drawPaths(ctx, size, paths) {
        for (var i = 0; i < paths.length; ++i) {
            var path = paths[i];
            if (path.length < 2) {
                path = [path[0], [path[0][0] + 0.001, path[0][1] + 0.001]];
            }
            ctx.beginPath();
            ctx.moveTo(path[0][0] * size, path[0][1] * size);
            for (var j = 1; j < path.length; ++j) {
                ctx.lineTo(path[j][0] * size, path[j][1] * size);
            }
            ctx.stroke();
        }
    }

    window.DrawingCell = DrawingCell;

})();
