const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
global.window = {};
vm.runInThisContext(fs.readFileSync('docs/predictor.js', 'utf8'));
const predictor = window.stuntingPredictor;

function manifest() {
    return JSON.parse(fs.readFileSync('docs/revision_manifest.json', 'utf8'));
}

for (const source of ['injected', 'fetched']) {
    for (const field of ['classes', 'gender', 'available', 'selected', 'path', 'features', 'artifactClasses']) {
        test(`${source} manifest rejects invalid ${field} before inference`, async () => {
            const data = manifest();
            if (field === 'classes') data.classes.reverse();
            if (field === 'gender') data.gender_encoding.Male = 0;
            if (field === 'available') data.available.haz = ['unknown'];
            if (field === 'selected') data.selected.classification.code = 'unknown';
            if (field === 'path') data.artifacts.classification_rf.browser_path = 'https://example.com/model.json';
            if (field === 'features') data.artifacts.classification_rf.features.reverse();
            if (field === 'artifactClasses') data.artifacts.classification_rf.classes.reverse();
            const originalFetch = global.fetch;
            global.fetch = async () => ({ok: true, json: async () => data});
            try {
                const engine = new StuntingPredictor(source === 'injected' ? {manifest: data} : {});
                await assert.rejects(engine.loadManifest(), /manifest/i);
            } finally { global.fetch = originalFetch; }
        });
    }
}

test('boosting margins produce a normalized model probability', () => {
    const model = {base_scores: [1000, 1001, 1000, 1000], trees: []};
    const result = predictor.predictClassification('xgb', model, {});
    assert.equal(result.classIndex, 1);
    assert.ok(Array.isArray(result.probabilities));
    assert.ok(Math.abs(result.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.ok(Math.abs(result.probabilities[1] - Math.E / (Math.E + 3)) < 1e-12);
});

test('XGBoost missing route resolves a child rather than treating node id as object', () => {
    const tree = {nodeid: 0, split: 'Height (cm)', split_condition: 70,
        yes: 1, no: 2, missing: 1,
        children: [{nodeid: 1, leaf: 2}, {nodeid: 2, leaf: -2}]};
    assert.equal(predictor.evalXgbRegNode(tree, {}), 2);
});

test('invalid user measurements fail before a model can supply a plausible result', async () => {
    await assert.rejects(predictor.predict({Gender_Encoded: 1, 'Age (months)': 12,
        'Height (cm)': -1, 'Weight (kg)': 8}, 'classification', 'rf'), /measurement/i);
});

test('concurrent slider requests share a model download', async () => {
    const originalFetch = global.fetch;
    let downloads = 0;
    const engine = new StuntingPredictor({manifest: manifest()});
    global.fetch = async () => { downloads++; return {ok: true, json: async () => [{leaf: [.1, .2, .6, .1]}]}; };
    try {
        await Promise.all([engine.loadModel('classification', 'rf'), engine.loadModel('classification', 'rf')]);
        assert.equal(downloads, 1);
    } finally { global.fetch = originalFetch; }
});

test('model downloads follow the active manifest instead of an old directory', async () => {
    const originalFetch = global.fetch;
    const requested = [];
    const engine = new StuntingPredictor({manifest: manifest()});
    global.fetch = async url => { requested.push(url); return {ok: true, json: async () => [{leaf: [1, 0, 0, 0]}]}; };
    try {
        await engine.loadModel('classification', 'rf');
        await engine.loadModel('classification', 'rf');
        assert.deepEqual(requested, ['models/paper_final/rf.json']);
    } finally { global.fetch = originalFetch; }
});

test('a failed model download can be retried successfully', async () => {
    const originalFetch = global.fetch;
    let downloads = 0;
    const engine = new StuntingPredictor({manifest: manifest()});
    global.fetch = async () => {
        downloads++;
        if (downloads === 1) throw new Error('Network interrupted');
        return {ok: true, json: async () => [{leaf: [1, 0, 0, 0]}]};
    };
    try {
        await assert.rejects(engine.loadModel('classification', 'rf'), /interrupted/);
        const model = await engine.loadModel('classification', 'rf');
        assert.equal(model[0].leaf[0], 1);
        assert.equal(downloads, 2);
    } finally { global.fetch = originalFetch; }
});

test('unknown model names are rejected before downloading a model', async () => {
    const originalFetch = global.fetch;
    let downloads = 0;
    const engine = new StuntingPredictor({manifest: manifest()});
    global.fetch = async () => { downloads++; throw new Error('Unexpected download'); };
    try {
        await assert.rejects(engine.loadModel('classification', 'unknown'), /Unsupported/);
        assert.equal(downloads, 0);
    } finally { global.fetch = originalFetch; }
});

test('a failed manifest request is retried rather than cached permanently', async () => {
    const originalFetch = global.fetch;
    let requests = 0;
    const engine = new StuntingPredictor();
    global.fetch = async () => {
        requests++;
        if (requests === 1) throw new Error('Manifest unavailable');
        return {ok: true, json: async () => manifest()};
    };
    try {
        assert.equal(typeof engine.loadManifest, 'function');
        await assert.rejects(engine.loadManifest(), /unavailable/);
        assert.equal((await engine.loadManifest()).classes[0], 'Normal');
        assert.equal(requests, 2);
    } finally { global.fetch = originalFetch; }
});

for (const code of ['linear', 'ridge']) {
    test(`${code} uses the recorded intercept and feature order`, () => {
        const data = {feature_names: ['Age (months)', 'Gender_Encoded', 'Height (cm)'],
            coefficients: [0.5, -2, 0.1], intercept: -3};
        assert.equal(predictor.predictRegression(code, data,
            {'Age (months)': 10, Gender_Encoded: 1, 'Height (cm)': 70}), 7);
    });
}

test('forest inference casts continuous measurements like native sklearn', () => {
    const tree = {split: 'Height (cm)', split_condition: 50.5,
        left: {leaf: [1, 0, 0, 0]}, right: {leaf: [0, 1, 0, 0]}};
    const features = {'Height (cm)': 50.5000001};
    assert.equal(predictor.predictClassification('rf', [tree], features).classIndex, 0);
    const regression = {...tree, left: {leaf: .4375}, right: {leaf: .5625}};
    assert.equal(predictor.predictRegression('rf', [regression], features), .4375);
});

test('CatBoost inference uses float32 measurements at split borders', () => {
    const tree = {splits: [{float_feature_index: 0, border: 50.5}], leaf_values: [.4375, .5625]};
    const features = {'Height (cm)': 50.5000001};
    assert.equal(predictor.evalCatBoostRegNode(tree, features, ['Height (cm)']), .4375);
    const classification = {...tree, leaf_values: [1, 0, 0, 0, 0, 1, 0, 0]};
    assert.deepEqual(predictor.evalCatBoostClassNode(classification, features, ['Height (cm)']), [1, 0, 0, 0]);
});
