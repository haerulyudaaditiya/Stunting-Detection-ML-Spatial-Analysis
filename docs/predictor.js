// Predictor Engine for Multi-Task Stunting Analysis
// Handles dynamic fetching of JSON models and inference for Classification, HAZ Regression, and WAZ Regression.

class StuntingPredictor {
    constructor() {
        this.models = {}; // Cache for downloaded models
        this.pendingModels = {};
        this.modelDirectory = 'models/revision_1';
        this.CLASSES = ['Normal', 'Severely Stunted', 'Stunted', 'Tall'];
    }

    async loadModel(task, modelName) {
        // Build filename: e.g. xgb.json for classification, xgb_haz.json for HAZ, xgb_waz.json for WAZ
        const filename = task === 'classification' ? `${modelName}.json` : `${modelName}_${task}.json`;
        const cacheKey = `${task}_${modelName}`;
        
        if (this.models[cacheKey]) {
            return this.models[cacheKey]; // Return cached model
        }
        if (this.pendingModels[cacheKey]) return this.pendingModels[cacheKey];

        console.log(`Downloading model: ${filename}...`);
        const pending = (async () => { try {
            const response = await fetch(`${this.modelDirectory}/${filename}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            this.models[cacheKey] = data;
            console.log(`Model ${filename} loaded successfully.`);
            return data;
        } catch (error) {
            console.error(`Error loading model ${filename}:`, error);
            throw error;
        } finally { delete this.pendingModels[cacheKey]; } })();
        this.pendingModels[cacheKey] = pending;
        return pending;
    }

    featureValue(features, name) {
        const aliases = {
            'Age_(months)': 'Age (months)', 'Height_(cm)': 'Height (cm)',
            'Weight_(kg)': 'Weight (kg)', 'Umur_(bulan)': 'Umur (bulan)',
            'Tinggi_Badan_(cm)': 'Tinggi Badan (cm)', 'Berat_Badan_(kg)': 'Berat Badan (kg)',
            'Jenis_Kelamin': 'Jenis Kelamin'
        };
        return features[aliases[name] || name];
    }

    // --- CLASSIFICATION INFERENCE LOGIC --- //
    
    predictClassification(modelName, modelData, features) {
        let scores = [0, 0, 0, 0];
        if (modelName === 'rf' || modelName === 'et') {
            const numTrees = modelData.length;
            for (let i = 0; i < numTrees; i++) {
                const classScores = this.evalSklearnClassNode(modelData[i], features);
                for (let j = 0; j < 4; j++) scores[j] += classScores[j] / numTrees;
            }
        } else if (modelName === 'xgb') {
            const base = modelData.base_scores || modelData.base_score;
            scores = Array.isArray(base) ? [...base] : [base, base, base, base]; // XGB base_score
            // XGB multi-class trees cycle through classes: tree i → class (i % num_classes)
            for (let i = 0; i < modelData.trees.length; i++) {
                const tree = modelData.trees[i];
                const classIdx = tree.tree_info !== undefined ? tree.tree_info : (i % 4);
                scores[classIdx] += this.evalXgbClassNode(tree, features);
            }
        } else if (modelName === 'lgbm') {
            for (const tree of modelData.tree_info) {
                scores[tree.tree_index % 4] += this.evalLgbmClassNode(tree.tree_structure, features, modelData.feature_names);
            }
        } else if (modelName === 'cb') {
            const cbFeatures = modelData.features_info.float_features.map(f => f.feature_id);
            for (const tree of modelData.oblivious_trees) {
                const leafVals = this.evalCatBoostClassNode(tree, features, cbFeatures);
                for (let j = 0; j < 4; j++) scores[j] += leafVals[j];
            }
            const scale = modelData.scale_and_bias?.[0] ?? 1;
            const bias = modelData.scale_and_bias?.[1] ?? [0, 0, 0, 0];
            scores = scores.map((value, i) => scale * value + (bias[i] || 0));
        } else {
            throw new Error(`Unsupported classification model: ${modelName}`);
        }
        
        // Find argmax
        let maxIdx = 0;
        for (let i = 1; i < 4; i++) {
            if (scores[i] > scores[maxIdx]) maxIdx = i;
        }
        let probabilities;
        if (modelName === 'rf' || modelName === 'et') {
            const total = scores.reduce((sum, value) => sum + value, 0);
            probabilities = scores.map(value => value / total);
        } else {
            const maximum = Math.max(...scores);
            const exponentials = scores.map(value => Math.exp(value - maximum));
            const total = exponentials.reduce((sum, value) => sum + value, 0);
            probabilities = exponentials.map(value => value / total);
        }
        if (probabilities.some(value => !Number.isFinite(value))) throw new Error('Nonfinite model probability');
        return {
            classIndex: maxIdx,
            className: this.CLASSES[maxIdx],
            scores: scores,
            probabilities: probabilities
        };
    }

    evalSklearnClassNode(node, features) {
        if (node.leaf !== undefined) return node.leaf;
        const val = this.featureValue(features, node.split);
        if (Math.fround(val) <= node.split_condition) {
            return this.evalSklearnClassNode(node.left !== undefined ? node.left : node.yes, features);
        } else {
            return this.evalSklearnClassNode(node.right !== undefined ? node.right : node.no, features);
        }
    }

    evalXgbClassNode(node, features) {
        if (node.leaf !== undefined) return node.leaf;
        const val = this.featureValue(features, node.split);
        if (val === undefined || val === null) {
            return this.evalXgbClassNode(node.children.find(c => c.nodeid === node.missing), features);
        }
        const children = node.children;
        if (Math.fround(val) < Math.fround(node.split_condition)) {
            return this.evalXgbClassNode(children.find(c => c.nodeid === node.yes), features);
        } else {
            return this.evalXgbClassNode(children.find(c => c.nodeid === node.no), features);
        }
    }

    evalLgbmClassNode(node, features, featureNames) {
        if (node.leaf_value !== undefined) return node.leaf_value;
        let fname = featureNames[node.split_feature];
        if (fname === 'Tinggi_Badan_(cm)') fname = 'Tinggi Badan (cm)';
        if (fname === 'Umur_(bulan)') fname = 'Umur (bulan)';
        if (fname === 'Berat_Badan_(kg)') fname = 'Berat Badan (kg)';
        if (fname === 'Jenis_Kelamin') fname = 'Jenis Kelamin';
        
        const val = this.featureValue(features, fname);
        if (val === undefined || val === null) {
            return this.evalLgbmClassNode(node.default_left ? node.left_child : node.right_child, features, featureNames);
        }
        if (val <= node.threshold) {
            return this.evalLgbmClassNode(node.left_child, features, featureNames);
        } else {
            return this.evalLgbmClassNode(node.right_child, features, featureNames);
        }
    }

    evalCatBoostClassNode(tree, features, cbFeatures) {
        let nodeIdx = 0;
        for (let depth = 0; depth < tree.splits.length; depth++) {
            const split = tree.splits[depth];
            const featureIdx = split.float_feature_index;
            const fname = cbFeatures[featureIdx];
            const val = this.featureValue(features, fname);
            const isTrue = Math.fround(val) > split.border;
            if (isTrue) {
                nodeIdx |= (1 << depth);
            }
        }
        // Multi-class: leaf_values are interleaved [c0_l0, c1_l0, c2_l0, c3_l0, c0_l1, ...]
        const base = nodeIdx * 4;
        return tree.leaf_values.slice(base, base + 4);
    }

    // --- REGRESSION INFERENCE LOGIC --- //

    predictRegression(modelName, modelData, features) {
        let sum = 0;
        if (modelName === 'rf' || modelName === 'et') {
            for (const tree of modelData) {
                sum += this.evalSklearnRegNode(tree, features);
            }
            sum /= modelData.length;
        } else if (modelName === 'xgb') {
            sum = Array.isArray(modelData.base_score) ? modelData.base_score[0] : modelData.base_score;
            for (const tree of modelData.trees) {
                sum += this.evalXgbRegNode(tree, features);
            }
        } else if (modelName === 'lgbm') {
            const featureNames = modelData.feature_names;
            for (const tree of modelData.tree_info) {
                sum += this.evalLgbmRegNode(tree.tree_structure, features, featureNames);
            }
        } else if (modelName === 'cb') {
            const cbFeatures = modelData.features_info.float_features.map(f => f.feature_id);
            for (const tree of modelData.oblivious_trees) {
                sum += this.evalCatBoostRegNode(tree, features, cbFeatures);
            }
            sum = (modelData.scale_and_bias?.[0] ?? 1) * sum + (modelData.scale_and_bias?.[1]?.[0] ?? 0);
        } else if (modelName === 'linear' || modelName === 'ridge') {
            sum = modelData.intercept;
            modelData.feature_names.forEach((name, i) => { sum += modelData.coefficients[i] * this.featureValue(features, name); });
        } else {
            throw new Error(`Unsupported regression model: ${modelName}`);
        }
        return sum;
    }

    evalSklearnRegNode(node, features) {
        if (node.leaf !== undefined) return node.leaf;
        const val = this.featureValue(features, node.split);
        if (Math.fround(val) <= node.split_condition) {
            return this.evalSklearnRegNode(node.left !== undefined ? node.left : node.yes, features);
        } else {
            return this.evalSklearnRegNode(node.right !== undefined ? node.right : node.no, features);
        }
    }

    evalXgbRegNode(node, features) {
        if (node.leaf !== undefined) return node.leaf;
        const val = this.featureValue(features, node.split);
        if (val === undefined || val === null) {
            return this.evalXgbRegNode(node.children.find(c => c.nodeid === node.missing), features);
        }
        const children = node.children;
        if (Math.fround(val) < Math.fround(node.split_condition)) {
            return this.evalXgbRegNode(children.find(c => c.nodeid === node.yes), features);
        } else {
            return this.evalXgbRegNode(children.find(c => c.nodeid === node.no), features);
        }
    }

    evalLgbmRegNode(node, features, featureNames) {
        if (node.leaf_value !== undefined) return node.leaf_value;
        let fname = featureNames[node.split_feature];
        if (fname === 'Tinggi_Badan_(cm)') fname = 'Tinggi Badan (cm)';
        if (fname === 'Umur_(bulan)') fname = 'Umur (bulan)';
        if (fname === 'Berat_Badan_(kg)') fname = 'Berat Badan (kg)';
        if (fname === 'Jenis_Kelamin') fname = 'Jenis Kelamin';
        
        const val = this.featureValue(features, fname);
        if (val === undefined || val === null) {
            return this.evalLgbmRegNode(node.default_left ? node.left_child : node.right_child, features, featureNames);
        }
        if (val <= node.threshold) {
            return this.evalLgbmRegNode(node.left_child, features, featureNames);
        } else {
            return this.evalLgbmRegNode(node.right_child, features, featureNames);
        }
    }

    evalCatBoostRegNode(tree, features, cbFeatures) {
        let nodeIdx = 0;
        for (let depth = 0; depth < tree.splits.length; depth++) {
            const split = tree.splits[depth];
            const featureIdx = split.float_feature_index;
            const fname = cbFeatures[featureIdx];
            const val = this.featureValue(features, fname);
            const isTrue = Math.fround(val) > split.border;
            if (isTrue) {
                nodeIdx |= (1 << depth);
            }
        }
        return tree.leaf_values[nodeIdx];
    }

    // Main entry point for the UI
    async predict(features, task, modelName) {
        if (!['classification', 'haz', 'waz'].includes(task)) throw new Error(`Unsupported task: ${task}`);
        const age = features['Age (months)'] ?? features['Umur (bulan)'];
        const height = features['Height (cm)'] ?? features['Tinggi Badan (cm)'];
        const weight = features['Weight (kg)'] ?? features['Berat Badan (kg)'];
        if (![0, 1].includes(features.Gender_Encoded) || !Number.isInteger(age) || age < 0 || age > 24 ||
            (task !== 'waz' && (!Number.isFinite(height) || height <= 0)) ||
            (task !== 'haz' && (!Number.isFinite(weight) || weight <= 0))) {
            throw new Error('Invalid measurement: age 0–24 months, encoded gender and positive sizes are required');
        }
        const modelData = await this.loadModel(task, modelName);
        if (task === 'classification') {
            return this.predictClassification(modelName, modelData, features);
        } else {
            return this.predictRegression(modelName, modelData, features);
        }
    }
}

// Instantiate globally so index.html can use it
window.stuntingPredictor = new StuntingPredictor();
