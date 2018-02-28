(function() {

    CELL_SIZE = 100;
    NUM_CLASSES = 3;
    IMAGE_SIZE = 28;

    function UI() {
        this.element = document.createElement('div');
        this.element.className = 'few-shot-container';

        this._cells = [];
        this._evaluator = null;
        this._predictions = new Predictions(NUM_CLASSES);

        this._trainElement = document.createElement('div');
        this._trainElement.className = 'few-shot-container-train-data';
        for (var i = 0; i < NUM_CLASSES; ++i) {
            var cell = new DrawingCell(['A', 'B', 'C', 'D', 'E'][i], CELL_SIZE);
            this._cells.push(cell);
            this._trainElement.appendChild(cell.element);
        }
        this.element.appendChild(this._trainElement);

        this._testAndControls = document.createElement('div');
        this._testAndControls.className = 'few-shot-container-test-and-controls';

        var testCell = new DrawingCell('?', CELL_SIZE);
        this._cells.push(testCell);
        this._testAndControls.appendChild(testCell.element);

        var controls = document.createElement('div');
        controls.className = 'few-shot-container-controls';

        var clearButton = document.createElement('button');
        clearButton.className = 'few-shot-container-clear';
        clearButton.textContent = 'Clear';
        clearButton.addEventListener('click', this._clear.bind(this));
        controls.appendChild(clearButton);

        var clearAllButton = document.createElement('button');
        clearAllButton.className = 'few-shot-container-clear-all';
        clearAllButton.textContent = 'Clear All';
        clearAllButton.addEventListener('click', this._clearAll.bind(this));
        controls.appendChild(clearAllButton);

        this._testAndControls.appendChild(controls);
        this.element.appendChild(this._testAndControls);

        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].onChange = this._cellChanged.bind(this);
        }

        this.element.appendChild(this._predictions.element);

        this._loadDefault();
    }

    UI.prototype._loadDefault = function() {
        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].setPaths(DEFAULT_PATHS[i]);
        }
        this._predictions.setProbs(DEFAULT_PROBS);
    };

    UI.prototype._clear = function() {
        this._cells[this._cells.length - 1].clear();
        this._cellChanged();
    };

    UI.prototype._clearAll = function() {
        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].clear();
        }
        this._cellChanged();
    };

    UI.prototype._cellChanged = function() {
        if (!this._hasEmptyCell()) {
            this._runNetwork();
        } else {
            this._predictions.setEnabled(false);
        }
    };

    UI.prototype._gotResult = function(obj) {
        console.log(obj);
        this._predictions.setEnabled(true);
        this._predictions.setProbs(obj.probs);
    };

    UI.prototype._hasEmptyCell = function() {
        for (var i = 0; i < this._cells.length; ++i) {
            if (this._cells[i].empty()) {
                return true;
            }
        }
        return false;
    };

    UI.prototype._runNetwork = function() {
        if (!this._evaluator) {
            this._evaluator = new Evaluator();
            this._evaluator.onResult = this._gotResult.bind(this);
        }
        var joined = [];
        for (var i = 0; i < this._cells.length; ++i) {
            var data = this._cells[i].tensor(IMAGE_SIZE).data;
            for (var j = 0; j < data.length; ++j) {
                joined.push(data[j]);
            }
        }
        this._evaluator.evaluate({data: joined, classes: NUM_CLASSES});
    };

    window.UI = UI;

})();
