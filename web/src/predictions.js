(function() {

    var EXTRA_SPACE = 90;
    var LABEL_WIDTH = 20;

    function Predictions(classes) {
        this.classes = classes;
        this.element = document.createElement('div');
        this.element.className = 'few-shot-predictions';

        this._rows = [];
        this._bars = [];
        this._percentLabels = [];

        for (var i = 0; i < classes; ++i) {
            var barContainer = document.createElement('div');
            barContainer.className = 'few-shot-predictions-bar-container';
            var bar = document.createElement('div');
            bar.className = 'few-shot-predictions-bar';
            barContainer.appendChild(bar);
            var percentLabel = document.createElement('label');
            percentLabel.className = 'few-shot-predictions-percent-label';

            var row = document.createElement('div');
            row.className = 'few-shot-predictions-row';
            row.appendChild(barContainer);
            row.appendChild(percentLabel);
            this.element.appendChild(row);

            this._rows.push(row);
            this._bars.push(bar);
            this._percentLabels.push(percentLabel);
        }
    }

    Predictions.prototype.setEnabled = function(enabled) {
        if (enabled) {
            this.element.className = 'few-shot-predictions';
        } else {
            this.element.className = 'few-shot-predictions few-shot-predictions-disabled';
        }
    };

    Predictions.prototype.setProbs = function(probs) {
        var max = Math.max.apply(Math, probs);
        for (var i = 0; i < this._bars.length; ++i) {
            this._bars[i].style.width = (probs[i] * 100).toFixed(2) + '%';
            this._percentLabels[i].textContent = (probs[i] * 100).toFixed(1) + '%';
            if (probs[i] !== max) {
                this._rows[i].className = 'few-shot-predictions-row';
            } else {
                this._rows[i].className = 'few-shot-predictions-row ' +
                    'few-shot-predictions-row-active';
            }
        }
    };

    window.Predictions = Predictions;

})();
