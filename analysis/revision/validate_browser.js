// Check the actual exported JavaScript artifacts against native Python predictions.
// Run from repository root: node analysis/revision/validate_browser.js
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
global.window = {};
vm.runInThisContext(fs.readFileSync('docs/predictor.js', 'utf8'));
const manifest = JSON.parse(fs.readFileSync('docs/revision_manifest.json', 'utf8'));
const directory = path.join('docs', manifest.model_directory);
const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'parity_samples.json'), 'utf8'));
const report = {checked_at_utc: new Date().toISOString(), tolerance: 1e-5,
    predictor_sha256: crypto.createHash('sha256').update(fs.readFileSync('docs/predictor.js')).digest('hex'),
    checks: [], total_predictions: 0};
(async () => {
    for (const fixture of fixtures) {
        const task = fixture.task.toLowerCase();
        const filename = task === 'classification' ? `${fixture.code}.json` : `${fixture.code}_${task}.json`;
        const source = fs.readFileSync(path.join(directory, filename));
        const engine = new StuntingPredictor({manifest});
        engine.models[`${task}_${fixture.code}`] = JSON.parse(source);
        let maximumError = 0;
        for (let i = 0; i < fixture.features.length; i++) {
            const result = await engine.predict(fixture.features[i], task, fixture.code);
            if (task === 'classification') {
                assert.equal(result.classIndex, fixture.prediction[i], `${filename} class at row ${i}`);
                const sourceClasses = ['Normal', 'Severely Stunted', 'Stunted', 'Tall'];
                assert.equal(result.className, sourceClasses[fixture.prediction[i]], `${filename} label at row ${i}`);
                for (let c = 0; c < 4; c++) {
                    const error = Math.abs(result.probabilities[c] - fixture.probabilities[i][c]);
                    maximumError = Math.max(maximumError, error);
                    assert.ok(error <= report.tolerance, `${filename} probability error ${error} at row ${i}`);
                }
            } else {
                const error = Math.abs(result - fixture.prediction[i]);
                maximumError = Math.max(maximumError, error);
                assert.ok(error <= report.tolerance, `${filename} regression error ${error} at row ${i}`);
            }
        }
        report.checks.push({filename, predictions: fixture.features.length, maximum_absolute_error: maximumError,
            sha256: crypto.createHash('sha256').update(source).digest('hex')});
        report.total_predictions += fixture.features.length;
        console.log(`${filename}: ${fixture.features.length} predictions, max error ${maximumError}`);
    }
    const output = 'scratch/dashboard_validation';
    fs.mkdirSync(output, {recursive: true});
    fs.writeFileSync(path.join(output, 'browser_parity.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(`PASS: ${report.total_predictions} predictions across ${report.checks.length} models`);
})().catch(error => { console.error(error); process.exitCode = 1; });
