// tree.rs — Decision tree inference, random forest, gradient boosted trees, feature importance
use crate::tensor::Tensor;

/// Split condition in a decision tree node
#[derive(Clone, Debug)]
pub enum SplitCondition {
    /// feature[index] <= threshold -> go left
    LessOrEqual { feature_idx: usize, threshold: f64 },
    /// feature[index] is in set -> go left
    InSet { feature_idx: usize, values: Vec<u32> },
    /// For missing value handling
    IsNan { feature_idx: usize },
}

impl SplitCondition {
    pub fn evaluate(&self, features: &[f64]) -> bool {
        match self {
            SplitCondition::LessOrEqual { feature_idx, threshold } => {
                features[*feature_idx] <= *threshold
            }
            SplitCondition::InSet { feature_idx, values } => {
                let v = features[*feature_idx] as u32;
                values.contains(&v)
            }
            SplitCondition::IsNan { feature_idx } => {
                features[*feature_idx].is_nan()
            }
        }
    }

    pub fn feature_index(&self) -> usize {
        match self {
            SplitCondition::LessOrEqual { feature_idx, .. } => *feature_idx,
            SplitCondition::InSet { feature_idx, .. } => *feature_idx,
            SplitCondition::IsNan { feature_idx } => *feature_idx,
        }
    }
}

/// A single node in the decision tree (stored in a flat array)
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub split: Option<SplitCondition>,
    pub left_child: usize,   // index in nodes array (0 = no child for leaves)
    pub right_child: usize,
    pub leaf_value: f64,
    pub is_leaf: bool,
    pub depth: usize,
    pub num_samples: usize,  // how many training samples reached this node
    pub impurity: f64,
}

impl TreeNode {
    pub fn leaf(value: f64, depth: usize, num_samples: usize) -> Self {
        Self {
            split: None, left_child: 0, right_child: 0,
            leaf_value: value, is_leaf: true, depth, num_samples, impurity: 0.0,
        }
    }

    pub fn internal(split: SplitCondition, left: usize, right: usize, depth: usize, num_samples: usize, impurity: f64) -> Self {
        Self {
            split: Some(split), left_child: left, right_child: right,
            leaf_value: 0.0, is_leaf: false, depth, num_samples, impurity,
        }
    }
}

/// Decision tree for inference (pre-built)
#[derive(Clone, Debug)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
    pub num_features: usize,
    pub num_classes: Option<usize>, // None for regression
    pub default_direction: Vec<bool>, // for missing values: true=left, false=right
}

impl DecisionTree {
    pub fn new(nodes: Vec<TreeNode>, num_features: usize) -> Self {
        let n = nodes.len();
        Self {
            nodes, num_features, num_classes: None,
            default_direction: vec![true; n],
        }
    }

    pub fn with_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = Some(num_classes);
        self
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        let mut node_idx = 0usize;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.leaf_value;
            }
            let split = node.split.as_ref().unwrap();
            let fi = split.feature_index();
            let go_left = if features[fi].is_nan() {
                self.default_direction[node_idx]
            } else {
                split.evaluate(features)
            };
            node_idx = if go_left { node.left_child } else { node.right_child };
        }
    }

    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.predict_single(f)).collect()
    }

    pub fn predict_tensor(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let n = x.shape[0];
        let feat = x.shape[1];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let row = &x.data[i * feat..(i + 1) * feat];
            results.push(self.predict_single(row));
        }
        Tensor::from_vec(results, &[n])
    }

    pub fn leaf_path(&self, features: &[f64]) -> Vec<usize> {
        let mut path = Vec::new();
        let mut node_idx = 0;
        loop {
            path.push(node_idx);
            let node = &self.nodes[node_idx];
            if node.is_leaf { break; }
            let split = node.split.as_ref().unwrap();
            let fi = split.feature_index();
            let go_left = if features[fi].is_nan() {
                self.default_direction[node_idx]
            } else {
                split.evaluate(features)
            };
            node_idx = if go_left { node.left_child } else { node.right_child };
        }
        path
    }

    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf).count()
    }

    pub fn num_internal(&self) -> usize {
        self.nodes.iter().filter(|n| !n.is_leaf).count()
    }

    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let mut importance = vec![0.0; self.num_features];
        for node in &self.nodes {
            if !node.is_leaf {
                let fi = node.split.as_ref().unwrap().feature_index();
                let left = &self.nodes[node.left_child];
                let right = &self.nodes[node.right_child];
                let n = node.num_samples as f64;
                let nl = left.num_samples as f64;
                let nr = right.num_samples as f64;
                if n > 0.0 {
                    let gain = node.impurity - (nl / n) * left.impurity - (nr / n) * right.impurity;
                    importance[fi] += gain.max(0.0) * n;
                }
            }
        }
        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            for v in importance.iter_mut() { *v /= total; }
        }
        importance
    }

    pub fn feature_importance_split_count(&self) -> Vec<f64> {
        let mut counts = vec![0.0; self.num_features];
        for node in &self.nodes {
            if !node.is_leaf {
                let fi = node.split.as_ref().unwrap().feature_index();
                counts[fi] += 1.0;
            }
        }
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            for v in counts.iter_mut() { *v /= total; }
        }
        counts
    }

    /// Build a simple decision stump (1-level tree) for a feature
    pub fn stump(feature_idx: usize, threshold: f64, left_val: f64, right_val: f64, num_features: usize) -> Self {
        let root = TreeNode::internal(
            SplitCondition::LessOrEqual { feature_idx, threshold },
            1, 2, 0, 100, 0.5,
        );
        let left = TreeNode::leaf(left_val, 1, 50);
        let right = TreeNode::leaf(right_val, 1, 50);
        Self::new(vec![root, left, right], num_features)
    }

    /// Build a balanced binary tree with given depth (for testing)
    pub fn balanced(depth: usize, num_features: usize) -> Self {
        let mut nodes = Vec::new();
        let mut rng_state = 12345u64;
        fn build(nodes: &mut Vec<TreeNode>, depth: usize, cur_depth: usize, num_feat: usize, rng: &mut u64) -> usize {
            let idx = nodes.len();
            if cur_depth >= depth {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = (*rng >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0;
                nodes.push(TreeNode::leaf(val, cur_depth, 10));
                return idx;
            }
            nodes.push(TreeNode::leaf(0.0, 0, 0)); // placeholder
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let fi = (*rng >> 32) as usize % num_feat;
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let thresh = (*rng >> 32) as f64 / u32::MAX as f64;
            let left = build(nodes, depth, cur_depth + 1, num_feat, rng);
            let right = build(nodes, depth, cur_depth + 1, num_feat, rng);
            nodes[idx] = TreeNode::internal(
                SplitCondition::LessOrEqual { feature_idx: fi, threshold: thresh },
                left, right, cur_depth, 100 >> cur_depth, 0.5 / (cur_depth + 1) as f64,
            );
            idx
        }
        build(&mut nodes, depth, 0, num_features, &mut rng_state);
        Self::new(nodes, num_features)
    }
}

/// Multi-output decision tree (leaf stores a vector)
#[derive(Clone, Debug)]
pub struct MultiOutputTree {
    pub nodes: Vec<TreeNode>,
    pub leaf_values: Vec<Vec<f64>>, // indexed by node index
    pub num_features: usize,
    pub num_outputs: usize,
}

impl MultiOutputTree {
    pub fn predict_single(&self, features: &[f64]) -> Vec<f64> {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return self.leaf_values[node_idx].clone();
            }
            let go_left = node.split.as_ref().unwrap().evaluate(features);
            node_idx = if go_left { node.left_child } else { node.right_child };
        }
    }
}

/// Random forest: ensemble of decision trees
#[derive(Clone, Debug)]
pub struct RandomForest {
    pub trees: Vec<DecisionTree>,
    pub num_features: usize,
    pub is_classifier: bool,
    pub num_classes: Option<usize>,
}

impl RandomForest {
    pub fn new(trees: Vec<DecisionTree>, is_classifier: bool) -> Self {
        let nf = trees[0].num_features;
        let nc = if is_classifier { trees[0].num_classes } else { None };
        Self { trees, num_features: nf, is_classifier, num_classes: nc }
    }

    /// Generate a test forest with stumps
    pub fn test_forest(num_trees: usize, num_features: usize, depth: usize) -> Self {
        let trees: Vec<DecisionTree> = (0..num_trees)
            .map(|_| DecisionTree::balanced(depth, num_features))
            .collect();
        Self::new(trees, false)
    }

    /// Predict by averaging tree outputs (regression)
    pub fn predict_single(&self, features: &[f64]) -> f64 {
        if self.is_classifier {
            self.predict_class(features) as f64
        } else {
            let sum: f64 = self.trees.iter().map(|t| t.predict_single(features)).sum();
            sum / self.trees.len() as f64
        }
    }

    /// Classification: majority vote
    pub fn predict_class(&self, features: &[f64]) -> usize {
        let nc = self.num_classes.unwrap_or(2);
        let mut votes = vec![0usize; nc];
        for tree in &self.trees {
            let pred = tree.predict_single(features).round() as usize;
            if pred < nc { votes[pred] += 1; }
        }
        votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let nc = self.num_classes.unwrap_or(2);
        let mut votes = vec![0.0; nc];
        let n = self.trees.len() as f64;
        for tree in &self.trees {
            let pred = tree.predict_single(features);
            let cls = pred.round() as usize;
            if cls < nc { votes[cls] += 1.0 / n; }
        }
        votes
    }

    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.predict_single(f)).collect()
    }

    pub fn predict_tensor(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let n = x.shape[0];
        let feat = x.shape[1];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let row = &x.data[i * feat..(i + 1) * feat];
            results.push(self.predict_single(row));
        }
        Tensor::from_vec(results, &[n])
    }

    /// Parallel-ish prediction (chunked)
    pub fn predict_parallel(&self, features: &[Vec<f64>], chunk_size: usize) -> Vec<f64> {
        let n = features.len();
        let mut results = vec![0.0; n];
        for start in (0..n).step_by(chunk_size) {
            let end = (start + chunk_size).min(n);
            for i in start..end {
                results[i] = self.predict_single(&features[i]);
            }
        }
        results
    }

    pub fn feature_importance(&self) -> Vec<f64> {
        let mut total = vec![0.0; self.num_features];
        for tree in &self.trees {
            let imp = tree.feature_importance_gain();
            for (i, v) in imp.iter().enumerate() {
                total[i] += v;
            }
        }
        let sum: f64 = total.iter().sum();
        if sum > 0.0 { for v in total.iter_mut() { *v /= sum; } }
        total
    }

    pub fn num_trees(&self) -> usize { self.trees.len() }
    pub fn avg_depth(&self) -> f64 {
        self.trees.iter().map(|t| t.depth() as f64).sum::<f64>() / self.trees.len() as f64
    }
    pub fn avg_leaves(&self) -> f64 {
        self.trees.iter().map(|t| t.num_leaves() as f64).sum::<f64>() / self.trees.len() as f64
    }

    /// Out-of-bag style error estimate (not real OOB, just variance estimate)
    pub fn prediction_variance(&self, features: &[f64]) -> f64 {
        let preds: Vec<f64> = self.trees.iter().map(|t| t.predict_single(features)).collect();
        let mean = preds.iter().sum::<f64>() / preds.len() as f64;
        preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>() / preds.len() as f64
    }

    /// Prediction intervals using tree variance
    pub fn prediction_interval(&self, features: &[f64], confidence: f64) -> (f64, f64, f64) {
        let preds: Vec<f64> = self.trees.iter().map(|t| t.predict_single(features)).collect();
        let mean = preds.iter().sum::<f64>() / preds.len() as f64;
        let var = preds.iter().map(|&p| (p - mean) * (p - mean)).sum::<f64>() / preds.len() as f64;
        let std = var.sqrt();
        let z = if confidence >= 0.99 { 2.576 } else if confidence >= 0.95 { 1.96 } else { 1.645 };
        (mean - z * std, mean, mean + z * std)
    }
}

/// Gradient Boosted Tree ensemble
#[derive(Clone, Debug)]
pub struct GradientBoostedTrees {
    pub trees: Vec<DecisionTree>,
    pub learning_rates: Vec<f64>,
    pub initial_prediction: f64,
    pub num_features: usize,
    pub loss_type: GBTLoss,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GBTLoss {
    SquaredError,
    AbsoluteError,
    Huber(f64), // delta
    LogLoss,    // binary classification
    Quantile(f64), // quantile regression
}

impl GradientBoostedTrees {
    pub fn new(trees: Vec<DecisionTree>, learning_rates: Vec<f64>, initial: f64, loss: GBTLoss) -> Self {
        assert_eq!(trees.len(), learning_rates.len());
        let nf = trees[0].num_features;
        Self { trees, learning_rates, initial_prediction: initial, num_features: nf, loss_type: loss }
    }

    pub fn test_gbt(num_trees: usize, depth: usize, num_features: usize, lr: f64) -> Self {
        let trees: Vec<DecisionTree> = (0..num_trees)
            .map(|_| DecisionTree::balanced(depth, num_features))
            .collect();
        let lrs = vec![lr; num_trees];
        Self::new(trees, lrs, 0.0, GBTLoss::SquaredError)
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for (tree, &lr) in self.trees.iter().zip(self.learning_rates.iter()) {
            pred += lr * tree.predict_single(features);
        }
        match self.loss_type {
            GBTLoss::LogLoss => 1.0 / (1.0 + (-pred).exp()), // sigmoid
            _ => pred,
        }
    }

    /// Raw score before any link function
    pub fn raw_score(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for (tree, &lr) in self.trees.iter().zip(self.learning_rates.iter()) {
            pred += lr * tree.predict_single(features);
        }
        pred
    }

    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.predict_single(f)).collect()
    }

    pub fn predict_tensor(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let n = x.shape[0];
        let feat = x.shape[1];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(self.predict_single(&x.data[i * feat..(i + 1) * feat]));
        }
        Tensor::from_vec(results, &[n])
    }

    /// Staged prediction: returns cumulative prediction after each tree
    pub fn staged_predict(&self, features: &[f64]) -> Vec<f64> {
        let mut pred = self.initial_prediction;
        let mut results = Vec::with_capacity(self.trees.len());
        for (tree, &lr) in self.trees.iter().zip(self.learning_rates.iter()) {
            pred += lr * tree.predict_single(features);
            let out = match self.loss_type {
                GBTLoss::LogLoss => 1.0 / (1.0 + (-pred).exp()),
                _ => pred,
            };
            results.push(out);
        }
        results
    }

    pub fn feature_importance(&self) -> Vec<f64> {
        let mut total = vec![0.0; self.num_features];
        for (tree, &lr) in self.trees.iter().zip(self.learning_rates.iter()) {
            let imp = tree.feature_importance_gain();
            for (i, v) in imp.iter().enumerate() {
                total[i] += v * lr;
            }
        }
        let sum: f64 = total.iter().sum();
        if sum > 0.0 { for v in total.iter_mut() { *v /= sum; } }
        total
    }

    pub fn num_trees(&self) -> usize { self.trees.len() }

    /// Partial dependence for one feature (average over samples)
    pub fn partial_dependence(&self, feature_idx: usize, grid: &[f64], background: &[Vec<f64>]) -> Vec<f64> {
        grid.iter().map(|&g| {
            let mut sum = 0.0;
            for sample in background {
                let mut modified = sample.clone();
                modified[feature_idx] = g;
                sum += self.predict_single(&modified);
            }
            sum / background.len() as f64
        }).collect()
    }

    /// SHAP-like feature contribution (tree path based)
    pub fn feature_contributions(&self, features: &[f64]) -> Vec<f64> {
        let mut contribs = vec![0.0; self.num_features];
        for (tree, &lr) in self.trees.iter().zip(self.learning_rates.iter()) {
            let path = tree.leaf_path(features);
            for i in 0..path.len() - 1 {
                let node = &tree.nodes[path[i]];
                if let Some(ref split) = node.split {
                    let fi = split.feature_index();
                    let child_val = if path[i + 1] == node.left_child {
                        tree.nodes[node.left_child].leaf_value
                    } else {
                        tree.nodes[node.right_child].leaf_value
                    };
                    contribs[fi] += lr * child_val / path.len() as f64;
                }
            }
        }
        contribs
    }
}

/// Multi-class GBT (one-vs-rest)
#[derive(Clone, Debug)]
pub struct MultiClassGBT {
    pub class_models: Vec<GradientBoostedTrees>,
    pub num_classes: usize,
}

impl MultiClassGBT {
    pub fn new(class_models: Vec<GradientBoostedTrees>) -> Self {
        let nc = class_models.len();
        Self { class_models, num_classes: nc }
    }

    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = self.class_models.iter().map(|m| m.raw_score(features)).collect();
        // softmax
        let max_r = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_r: Vec<f64> = raw.iter().map(|&r| (r - max_r).exp()).collect();
        let sum: f64 = exp_r.iter().sum();
        exp_r.iter().map(|&e| e / sum).collect()
    }

    pub fn predict_class(&self, features: &[f64]) -> usize {
        let proba = self.predict_proba(features);
        proba.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

/// XGBoost-style tree with gain-based splitting
#[derive(Clone, Debug)]
pub struct XGBoostTree {
    pub tree: DecisionTree,
    pub reg_lambda: f64,
    pub reg_alpha: f64,
}

impl XGBoostTree {
    pub fn new(tree: DecisionTree, lambda: f64, alpha: f64) -> Self {
        Self { tree, reg_lambda: lambda, reg_alpha: alpha }
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        self.tree.predict_single(features)
    }
}

/// LightGBM-style leaf-wise tree
#[derive(Clone, Debug)]
pub struct LightGBMTree {
    pub tree: DecisionTree,
    pub num_leaves: usize,
    pub min_data_in_leaf: usize,
}

impl LightGBMTree {
    pub fn new(tree: DecisionTree, num_leaves: usize) -> Self {
        Self { tree, num_leaves, min_data_in_leaf: 20 }
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        self.tree.predict_single(features)
    }
}

/// Extremely Randomized Trees (Extra Trees) ensemble
#[derive(Clone, Debug)]
pub struct ExtraTreesEnsemble {
    pub forest: RandomForest,
}

impl ExtraTreesEnsemble {
    pub fn new(trees: Vec<DecisionTree>) -> Self {
        Self { forest: RandomForest::new(trees, false) }
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        self.forest.predict_single(features)
    }
}

/// Isolation Forest for anomaly detection
#[derive(Clone, Debug)]
pub struct IsolationForest {
    pub trees: Vec<DecisionTree>,
    pub sample_size: usize,
}

impl IsolationForest {
    pub fn new(trees: Vec<DecisionTree>, sample_size: usize) -> Self {
        Self { trees, sample_size }
    }

    fn c_factor(n: usize) -> f64 {
        if n <= 1 { return 0.0; }
        let n = n as f64;
        2.0 * (n.ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
    }

    pub fn anomaly_score(&self, features: &[f64]) -> f64 {
        let avg_path: f64 = self.trees.iter().map(|t| {
            t.leaf_path(features).len() as f64
        }).sum::<f64>() / self.trees.len() as f64;

        let c = Self::c_factor(self.sample_size);
        if c == 0.0 { return 0.5; }
        2.0f64.powf(-avg_path / c)
    }

    pub fn is_anomaly(&self, features: &[f64], threshold: f64) -> bool {
        self.anomaly_score(features) > threshold
    }

    pub fn score_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.anomaly_score(f)).collect()
    }
}

/// Quantile regression forest
#[derive(Clone, Debug)]
pub struct QuantileForest {
    pub trees: Vec<DecisionTree>,
    pub leaf_samples: Vec<Vec<Vec<f64>>>, // [tree][leaf_node_idx] -> sorted samples
}

impl QuantileForest {
    pub fn new(trees: Vec<DecisionTree>, leaf_samples: Vec<Vec<Vec<f64>>>) -> Self {
        Self { trees, leaf_samples }
    }

    pub fn predict_quantile(&self, features: &[f64], quantile: f64) -> f64 {
        let mut all_samples = Vec::new();
        for (ti, tree) in self.trees.iter().enumerate() {
            let path = tree.leaf_path(features);
            let leaf_idx = *path.last().unwrap();
            if ti < self.leaf_samples.len() && leaf_idx < self.leaf_samples[ti].len() {
                all_samples.extend_from_slice(&self.leaf_samples[ti][leaf_idx]);
            }
        }
        if all_samples.is_empty() { return 0.0; }
        all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((quantile * all_samples.len() as f64) as usize).min(all_samples.len() - 1);
        all_samples[idx]
    }

    pub fn predict_interval(&self, features: &[f64], lo: f64, hi: f64) -> (f64, f64) {
        (self.predict_quantile(features, lo), self.predict_quantile(features, hi))
    }
}

/// Feature importance via permutation
pub fn permutation_importance(
    model: &RandomForest,
    x: &[Vec<f64>],
    y: &[f64],
    num_repeats: usize,
) -> Vec<f64> {
    let n = x.len();
    let nf = model.num_features;
    let base_preds: Vec<f64> = x.iter().map(|f| model.predict_single(f)).collect();
    let base_mse: f64 = base_preds.iter().zip(y.iter())
        .map(|(p, t)| (p - t) * (p - t)).sum::<f64>() / n as f64;

    let mut importance = vec![0.0; nf];
    for fi in 0..nf {
        let mut total_imp = 0.0;
        for rep in 0..num_repeats {
            // create permuted data
            let mut permuted: Vec<Vec<f64>> = x.to_vec();
            // simple shuffle using LCG
            let mut state = (fi * 1000 + rep) as u64;
            for i in (1..n).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (state >> 32) as usize % (i + 1);
                let tmp = permuted[i][fi];
                permuted[i][fi] = permuted[j][fi];
                permuted[j][fi] = tmp;
            }
            let perm_preds: Vec<f64> = permuted.iter().map(|f| model.predict_single(f)).collect();
            let perm_mse: f64 = perm_preds.iter().zip(y.iter())
                .map(|(p, t)| (p - t) * (p - t)).sum::<f64>() / n as f64;
            total_imp += perm_mse - base_mse;
        }
        importance[fi] = total_imp / num_repeats as f64;
    }
    // normalize
    let sum: f64 = importance.iter().map(|v| v.abs()).sum();
    if sum > 0.0 { for v in importance.iter_mut() { *v /= sum; } }
    importance
}

/// Tree serialization to bytes
pub fn serialize_tree(tree: &DecisionTree) -> Vec<u8> {
    let mut bytes = Vec::new();
    // header: magic + num_nodes + num_features
    bytes.extend_from_slice(b"TREE");
    bytes.extend_from_slice(&(tree.nodes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(tree.num_features as u32).to_le_bytes());
    for node in &tree.nodes {
        bytes.push(if node.is_leaf { 1 } else { 0 });
        bytes.extend_from_slice(&node.leaf_value.to_le_bytes());
        bytes.extend_from_slice(&(node.left_child as u32).to_le_bytes());
        bytes.extend_from_slice(&(node.right_child as u32).to_le_bytes());
        bytes.extend_from_slice(&(node.depth as u16).to_le_bytes());
        bytes.extend_from_slice(&(node.num_samples as u32).to_le_bytes());
        bytes.extend_from_slice(&node.impurity.to_le_bytes());
        if let Some(ref split) = node.split {
            match split {
                SplitCondition::LessOrEqual { feature_idx, threshold } => {
                    bytes.push(0);
                    bytes.extend_from_slice(&(*feature_idx as u32).to_le_bytes());
                    bytes.extend_from_slice(&threshold.to_le_bytes());
                }
                SplitCondition::InSet { feature_idx, values } => {
                    bytes.push(1);
                    bytes.extend_from_slice(&(*feature_idx as u32).to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
                    for &v in values {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                SplitCondition::IsNan { feature_idx } => {
                    bytes.push(2);
                    bytes.extend_from_slice(&(*feature_idx as u32).to_le_bytes());
                }
            }
        } else {
            bytes.push(255);
        }
    }
    bytes
}

pub fn deserialize_tree(bytes: &[u8]) -> Option<DecisionTree> {
    if bytes.len() < 12 { return None; }
    if &bytes[0..4] != b"TREE" { return None; }
    let num_nodes = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let num_features = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let mut pos = 12;
    let mut nodes = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        if pos >= bytes.len() { return None; }
        let is_leaf = bytes[pos] == 1;
        pos += 1;
        let leaf_value = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let left = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let right = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let depth = u16::from_le_bytes(bytes[pos..pos + 2].try_into().ok()?) as usize;
        pos += 2;
        let num_samples = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let impurity = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?);
        pos += 8;
        let split_type = bytes[pos];
        pos += 1;
        let split = if split_type == 0 {
            let fi = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let thresh = f64::from_le_bytes(bytes[pos..pos + 8].try_into().ok()?);
            pos += 8;
            Some(SplitCondition::LessOrEqual { feature_idx: fi, threshold: thresh })
        } else if split_type == 1 {
            let fi = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let n_vals = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut vals = Vec::new();
            for _ in 0..n_vals {
                let v = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
                pos += 4;
                vals.push(v);
            }
            Some(SplitCondition::InSet { feature_idx: fi, values: vals })
        } else if split_type == 2 {
            let fi = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            Some(SplitCondition::IsNan { feature_idx: fi })
        } else {
            None
        };
        nodes.push(TreeNode {
            split, left_child: left, right_child: right,
            leaf_value, is_leaf, depth, num_samples, impurity,
        });
    }
    Some(DecisionTree::new(nodes, num_features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_stump() {
        let tree = DecisionTree::stump(0, 0.5, -1.0, 1.0, 2);
        assert_eq!(tree.predict_single(&[0.3, 0.0]), -1.0);
        assert_eq!(tree.predict_single(&[0.7, 0.0]), 1.0);
    }

    #[test]
    fn test_balanced_tree() {
        let tree = DecisionTree::balanced(3, 4);
        let pred = tree.predict_single(&[0.5, 0.5, 0.5, 0.5]);
        assert!(pred.is_finite());
        assert!(tree.depth() <= 3);
    }

    #[test]
    fn test_random_forest() {
        let rf = RandomForest::test_forest(10, 4, 3);
        let pred = rf.predict_single(&[0.5, 0.3, 0.8, 0.1]);
        assert!(pred.is_finite());
        let imp = rf.feature_importance();
        assert_eq!(imp.len(), 4);
    }

    #[test]
    fn test_gbt() {
        let gbt = GradientBoostedTrees::test_gbt(5, 3, 4, 0.1);
        let pred = gbt.predict_single(&[0.5, 0.3, 0.8, 0.1]);
        assert!(pred.is_finite());
        let staged = gbt.staged_predict(&[0.5, 0.3, 0.8, 0.1]);
        assert_eq!(staged.len(), 5);
    }

    #[test]
    fn test_serialization() {
        let tree = DecisionTree::stump(0, 0.5, -1.0, 1.0, 2);
        let bytes = serialize_tree(&tree);
        let tree2 = deserialize_tree(&bytes).unwrap();
        assert_eq!(tree2.predict_single(&[0.3, 0.0]), -1.0);
        assert_eq!(tree2.predict_single(&[0.7, 0.0]), 1.0);
    }

    #[test]
    fn test_isolation_forest() {
        let trees: Vec<DecisionTree> = (0..5).map(|_| DecisionTree::balanced(4, 3)).collect();
        let iso = IsolationForest::new(trees, 256);
        let score = iso.anomaly_score(&[0.5, 0.5, 0.5]);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_prediction_interval() {
        let rf = RandomForest::test_forest(20, 4, 3);
        let (lo, mid, hi) = rf.prediction_interval(&[0.5, 0.3, 0.8, 0.1], 0.95);
        assert!(lo <= mid);
        assert!(mid <= hi);
    }

    #[test]
    fn test_feature_importance() {
        let tree = DecisionTree::stump(0, 0.5, -1.0, 1.0, 3);
        let imp = tree.feature_importance_gain();
        assert_eq!(imp.len(), 3);
    }
}
