const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
global.window = {};
vm.runInThisContext(fs.readFileSync('docs/predictor.js', 'utf8'));
const predictor = window.stuntingPredictor;

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
    const engine = new StuntingPredictor();
    global.fetch = async () => { downloads++; return {ok: true, json: async () => [{leaf: [.1, .2, .6, .1]}]}; };
    try {
        await Promise.all([engine.loadModel('classification', 'rf'), engine.loadModel('classification', 'rf')]);
        assert.equal(downloads, 1);
    } finally { global.fetch = originalFetch; }
});

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
