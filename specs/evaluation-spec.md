# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = number of correct predictions / total predictions. A prediction is
correct when predictions[i] exactly equals ground_truth[i] (same index, exact
string match). Divide by the number of predictions.
```

---

**Step-by-step logic:**

```
1. If predictions is empty, return 0.0.
2. Zip predictions and ground_truth together.
3. Count the pairs where the two are exactly equal.
4. Divide that count by len(predictions) and return the float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There's nothing to score, and dividing by zero would crash —
0.0 is the safe, meaningful "no correct predictions out of nothing" value.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

idx0 interview == interview ✓
idx1 solo == solo ✓
idx2 panel != solo ✗
idx3 interview != narrative ✗

2 correct / 4 total = 0.5  → returns 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For class C, an episode counts as correct when its ground-truth label is C AND
the prediction for that episode is also C. Grouping is by ground-truth label,
not predicted label.
```

---

**What does "total" mean for a given class?**

```
The number of episodes whose GROUND-TRUTH label is this class — not the number
of predictions for it. This is the key decision: grouping by ground truth makes
this recall per class. Grouping by predicted label would compute something
different (precision-flavored) and is the classic bug.
```

---

**Step-by-step logic:**

```
1. Initialize a dict with {"correct": 0, "total": 0, "accuracy": 0.0} per label.
2. Loop over zipped (predicted, truth) pairs.
3. For each pair, if truth is a valid label, increment that class's total; if
   predicted == truth, also increment that class's correct.
4. After the loop, set each class's accuracy = correct / total, or 0.0 if total
   is 0.
5. Return the dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set that class's accuracy to 0.0 (avoids division by zero). It means "no
episodes of this class to score," which the report still displays cleanly.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.00
solo           1       2      0.50
panel          1       1      1.00
narrative      0       1      0.00
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
