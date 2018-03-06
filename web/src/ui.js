(function() {

    CELL_SIZE = 100;
    NUM_CLASSES = 3;
    IMAGE_SIZE = 28;

    function UI() {
        this.element = document.getElementsByClassName('few-shot-container')[0];

        this._cells = [];
        this._evaluator = null;

        this._setupTrainElement();
        this.element.appendChild(createSeparator());
        this._setupTestElement();

        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].onChange = this._cellChanged.bind(this);
        }

        this._loadDefault();
    }

    UI.prototype._setupTrainElement = function() {
        this._trainElement = document.createElement('div');
        this._trainElement.className = 'few-shot-container-train-data';
        this._trainElement.appendChild(createSectionHeading('Training Data'));

        for (var i = 0; i < NUM_CLASSES; ++i) {
            var cell = new DrawingCell(CELL_SIZE);
            this._cells.push(cell);
            this._trainElement.appendChild(cell.element);
        }

        this._predictions = new Predictions(NUM_CLASSES);
        this._trainElement.appendChild(this._predictions.element);

        this._clearAllButton = document.createElement('button');
        this._clearAllButton.className = 'few-shot-container-clear-all';
        this._clearAllButton.textContent = 'Edit All';
        this._clearAllButton.addEventListener('click', this._clearAll.bind(this));
        this._trainElement.appendChild(this._clearAllButton);

        this.element.appendChild(this._trainElement);
    };

    UI.prototype._setupTestElement = function() {
        this._testElement = document.createElement('div');
        this._testElement.className = 'few-shot-container-test-data';
        this._testElement.appendChild(createSectionHeading('Input'));

        var testCell = new DrawingCell(CELL_SIZE);
        this._cells.push(testCell);
        this._testElement.appendChild(testCell.element);

        this._clearButton = document.createElement('button');
        this._clearButton.className = 'few-shot-container-clear';
        this._clearButton.textContent = 'Edit';
        this._clearButton.addEventListener('click', this._clear.bind(this));
        this._testElement.appendChild(this._clearButton);

        this.element.appendChild(this._testElement);
    };

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
        for (var i = 0; i < this._cells.length - 1; ++i) {
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

        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].enableInteraction();
        }
        this._clearAllButton.textContent = 'Erase All';
        this._clearButton.textContent = 'Erase';
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

    window.onload = function() {
        new UI();
    };

    function createSectionHeading(title) {
        var heading = document.createElement('label');
        heading.className = 'few-shot-section-heading';
        heading.textContent = title;
        return heading;
    }

    function createSeparator() {
        var separator = document.createElement('div');
        separator.className = 'few-shot-separator';
        return separator;
    }

})();
