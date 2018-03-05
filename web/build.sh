if [ -d 'build' ]; then
    rm -r build
fi
mkdir build

for file in deps/jsnet.js deps/model.js src/webworker.js; do
    cat $file >>build/webworker.js
done

for file in src/drawing.js src/evaluator.js src/predictions.js src/ui.js src/default.js deps/jsnet.js; do
    cat $file >>build/app.js
done
echo >>build/app.js
echo -n 'var WORKER_DATA = "' >>build/app.js
cat build/webworker.js |
    tr '\n' '`' |
    sed -E 's/\\/\\\\/g' |
    sed -E 's/`/\\n/g' |
    sed -E 's/"/\\"/g' >>build/app.js
perl -pi -e 'chomp if eof' build/app.js
echo -n '";' >>build/app.js
