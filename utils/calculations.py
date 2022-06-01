def f1_scores_from_conf_mat(cm):
    f1_scores = []
    for i in range(cm.shape[0]):
        precision = cm[i, i] / sum(cm[:, i]) if sum(cm[:, i]) else 0
        recall = cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        f1_scores.append(f1_score)

    return f1_scores