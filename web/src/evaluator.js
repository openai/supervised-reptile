(function() {

    function Evaluator() {
        this.onResult = function() {};
        this._worker = new Worker(makeWorkerBlob(WORKER_DATA));
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

    function makeWorkerBlob(data) {
        // See https://stackoverflow.com/questions/10343913/how-to-create-a-web-worker-from-a-string
        window.URL = window.URL || window.webkitURL;
        var blob;
        try {
            blob = new Blob([data], {type: 'application/javascript'});
        } catch (e) { // Backwards-compatibility
            window.BlobBuilder = window.BlobBuilder || window.WebKitBlobBuilder || window.MozBlobBuilder;
            blob = new BlobBuilder();
            blob.append(data);
            blob = blob.getBlob();
        }
        return URL.createObjectURL(blob);
    }

    window.Evaluator = Evaluator;

})();
