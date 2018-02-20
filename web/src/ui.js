(function() {

    CELL_SIZE = 100;
    NUM_CLASSES = 2;
    IMAGE_SIZE = 28;

    function UI() {
        this.element = document.createElement('div');

        this._cells = [];
        this._evaluator = null;

        this._trainElement = document.createElement('div');
        for (var i = 0; i < NUM_CLASSES; ++i) {
            var cell = new DrawingCell(['A', 'B', 'C', 'D', 'E'][i], CELL_SIZE);
            cell.element.style.display = 'inline-block';
            cell.element.style.margin = '3px';
            this._cells.push(cell);
            this._trainElement.appendChild(cell.element);
        }
        this.element.appendChild(this._trainElement);

        this._testElement = document.createElement('div');
        var testCell = new DrawingCell('?', CELL_SIZE);
        this._cells.push(testCell);
        this._testElement.appendChild(testCell.element);
        this.element.appendChild(this._testElement);

        for (var i = 0; i < this._cells.length; ++i) {
            this._cells[i].onChange = this._cellChanged.bind(this);
        }
    }

    UI.prototype._cellChanged = function() {
        if (!this._hasEmptyCell()) {
            this._runNetwork();
        }
    };

    UI.prototype._gotResult = function(obj) {
        console.log(obj);
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
                joined.push(j);
            }
        }
        this._evaluator.evaluate({data: joined, classes: NUM_CLASSES});
    };

    window.UI = UI;

})();
