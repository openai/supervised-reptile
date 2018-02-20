if [ -d 'build' ]; then
    rm -r build
fi
mkdir build

for file in src/drawing.js src/evaluator.js src/predictions.js src/ui.js deps/jsnet.js; do
    cat $file >>build/app.js
done

for file in src/webworker.js deps/jsnet.js; do
    cat $file >>build/webworker.js
done
