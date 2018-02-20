(function() {

    var WEBWORKER_PATH = 'build/webworker.js';

    function Evaluator() {
        this.onResult = function() {};
        this._worker = new Worker(WEBWORKER_PATH);
        this._queuedJob = null;
        this._runningJob = false;
        this._worker.onmessage = function(msg) {
            this._runningJob = false;
            if (this._queuedJob) {
                this._queuedJob = null;
                this.evaluate(tensors);
            } else {
                this.onResult(msg.data);
            }
        }.bind(this);
    }

    Evaluator.prototype.evaluate = function(job) {
        if (this._runningJob) {
            this._queuedJob = job;
        } else {
            this._worker.postMessage(job);
        }
    };

    window.Evaluator = Evaluator;

})();
