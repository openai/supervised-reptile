(function() {

    var LABEL_SIZE = 20;
    var UPSAMPLE = 2;
    var LINE_WIDTH = 0.05;

    function DrawingCell(label, size) {
        this.onChange = function() {};

        this.label = label;
        this.size = size;

        this.element = document.createElement('div');
        this.element.style.backgroundColor = '#f0f0f0';
        this.element.style.borderRadius = '3px';
        this.element.style.position = 'relative';
        this.element.style.width = size + 'px';
        this.element.style.height = size + 'px';
        this.element.style.overflow = 'hidden';

        this._canvas = document.createElement('canvas');
        this._canvas.position = 'absolute';
        this._canvas.style.top = '0';
        this._canvas.style.left = '0';
        this._canvas.style.width = size + 'px';
        this._canvas.style.height = size + 'px';
        this._canvas.width = size * UPSAMPLE;
        this._canvas.height = size * UPSAMPLE;
        this.element.appendChild(this._canvas);

        this._label = document.createElement('label');
        this._label.style.backgroundColor = '#5588c0';
        this._label.style.color = 'white';
        this._label.style.position = 'absolute';
        this._label.style.top = '0';
        this._label.style.left = '0';
        this._label.style.width = LABEL_SIZE + 'px';
        this._label.style.height = LABEL_SIZE + 'px';
        this._label.style.lineHeight = LABEL_SIZE + 'px';
        this._label.style.textAlign = 'center';
        this._label.style.borderRadius = '3px 0';
        this._label.textContent = label;
        this.element.appendChild(this._label);

        this._emptyLabel = document.createElement('label');
        this._emptyLabel.style.position = 'absolute';
        this._emptyLabel.style.top = '0';
        this._emptyLabel.style.left = '0';
        this._emptyLabel.style.width = size + 'px';
        this._emptyLabel.style.height = size + 'px';
        this._emptyLabel.style.color = '#999';
        this._emptyLabel.style.fontSize = Math.floor(size / 6) + 'px';
        this._emptyLabel.style.lineHeight = size + 'px';
        this._emptyLabel.style.textAlign = 'center';
        this._emptyLabel.style.pointerEvents = 'none';
        this._emptyLabel.textContent = 'Draw Here';
        this.element.appendChild(this._emptyLabel);

        this._paths = [];
        this._redraw();

        this._registerMouse();
        if ('ontouchstart' in document.documentElement) {
            this._registerTouch();
        }
    }

    DrawingCell.prototype.empty = function() {
        return this._paths.length === 0;
    };

    DrawingCell.prototype.clear = function() {
        this._paths = [];
        this._redraw();
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
        drawPaths(ctx, size, this._paths);

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
        // TODO: this.
    };

    DrawingCell.prototype._mousePosition = function(e) {
        var rect = this._canvas.getBoundingClientRect();
        return [(e.clientX - rect.left) / this._canvas.offsetWidth,
            (e.clientY - rect.top) / this._canvas.offsetHeight];
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
