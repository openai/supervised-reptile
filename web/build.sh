if [ -d 'build' ]; then
    rm -r build
fi
mkdir build

for file in src/drawing.js src/evaluator.js src/predictions.js src/ui.js src/default.js deps/jsnet.js; do
    cat $file >>build/app.js
done

for file in deps/jsnet.js deps/model.js src/webworker.js; do
    cat $file >>build/webworker.js
done
