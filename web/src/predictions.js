(function() {

    var EXTRA_SPACE = 90;
    var LABEL_WIDTH = 20;

    function Predictions(classes, width) {
        this.classes = classes;
        this.width = width;
        this.element = document.createElement('div');
        this.element.style.width = width + 'px';
        this.element.style.textAlign = 'left';
        this.element.style.display = 'inline-block';

        this._bars = [];
        this._percentLabels = [];

        for (var i = 0; i < classes; ++i) {
            var label = document.createElement('label');
            label.style.display = 'inline-block';
            label.style.width = LABEL_WIDTH + 'px';
            label.textContent = ['A', 'B', 'C', 'D', 'E'][i];
            var barContainer = document.createElement('div');
            barContainer.style.backgroundColor = '#f0f0f0';
            barContainer.style.width = (width - EXTRA_SPACE) + 'px';
            barContainer.style.height = '5px';
            barContainer.style.marginBottom = '3px';
            barContainer.style.display = 'inline-block';
            var bar = document.createElement('div');
            bar.style.backgroundColor = '#65bcd4';
            bar.style.width = '0%';
            bar.style.height = '100%';
            barContainer.appendChild(bar);
            var percentLabel = document.createElement('label');
            percentLabel.style.marginLeft = '10px';

            var row = document.createElement('div');
            row.style.marginBottom = '5px';
            row.appendChild(label);
            row.appendChild(barContainer);
            row.appendChild(percentLabel);
            this.element.appendChild(row);

            this._bars.push(bar);
            this._percentLabels.push(percentLabel);
        }
    }

    Predictions.prototype.setProbs = function(probs) {
        for (var i = 0; i < this._bars.length; ++i) {
            this._bars[i].style.width = (probs[i] * 100).toFixed(2) + '%';
            this._percentLabels[i].textContent = (probs[i] * 100).toFixed(1) + '%';
        }
    };

    window.Predictions = Predictions;

})();
