(function() {

    CELL_SIZE = 100;
    NUM_CLASSES = 3;
    IMAGE_SIZE = 28;

    function UI() {
        this.element = document.createElement('div');
        this.element.style.textAlign = 'center';

        this._cells = [];
        this._evaluator = null;
        this._predictions = new Predictions(NUM_CLASSES, 200);

        this._trainElement = document.createElement('div');
        this._trainElement.style.textAlign = 'center';
        for (var i = 0; i < NUM_CLASSES; ++i) {
            var cell = new DrawingCell(['A', 'B', 'C', 'D', 'E'][i], CELL_SIZE);
            cell.element.style.display = 'inline-block';
            cell.element.style.margin = '3px';
            this._cells.push(cell);
            this._trainElement.appendChild(cell.element);
        }
        this.element.appendChild(this._trainElement);

        this._testAndControls = document.createElement('div');
        this._testAndControls.style.textAlign = 'center';
        this._testAndControls.style.marginBottom = '10px';
        var testCell = new DrawingCell('?', CELL_SIZE);
        testCell.element.style.display = 'inline-block';
        this._cells.push(testCell);
        this._testAndControls.appendChild(testCell.element);
        var controls = document.createElement('div');
        controls.style.width = CELL_SIZE + 'px';
        controls.style.height = CELL_SIZE + 'px';
        controls.style.display = 'inline-block';
        controls.style.position = 'relative';
        var clearButton = document.createElement('button');
        var clearAllButton = document.createElement('button');
        for (var i = 0; i < 2; ++i) {
            var button = [clearButton, clearAllButton][i];
            button.style.position = 'absolute';
            button.style.width = '90%';
            button.style.height = '30%';
            button.style.left = '5%';
            button.style.border = 'none';
            button.style.backgroundColor = '#65bcd4';
            button.style.color = 'white';
            button.style.fontSize = '15px';
            button.style.cursor = 'pointer';
        }
        clearButton.style.top = '15%';
        clearButton.textContent = 'Clear';
        clearButton.addEventListener('click', this._clear.bind(this));
        controls.appendChild(clearButton);
        clearAllButton.style.top = '55%';
        clearAllButton.textContent = 'Clear All';
        clearAllButton.addEventListener('click', this._clearAll.bind(this));
        controls.appendChild(clearAllButton);
        this._testAndControls.appendChild(controls);
        this.element.appendChild(this._testAndControls);

        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].onChange = this._cellChanged.bind(this);
        }

        this.element.appendChild(this._predictions.element);
    }

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
        }
    };

    UI.prototype._gotResult = function(obj) {
        console.log(obj);
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
