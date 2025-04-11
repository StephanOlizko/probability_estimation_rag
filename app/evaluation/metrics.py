def evaluate(results, ground_truth):
    def accuracy_at_k(result, truth):
        return int(any(item in truth for item in result))

    total = 0
    for r, gt in zip(results, ground_truth):
        total += accuracy_at_k(r, gt)
    acc = total / len(results)
    print(f"Accuracy@K: {acc:.2f}")
