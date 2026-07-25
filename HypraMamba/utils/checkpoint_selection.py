VALID_CHECKPOINT_METRICS = {'oa', 'aa', 'miou', 'kappa'}
VALID_CHECKPOINT_TIE_BREAKS = {'latest', 'earliest', 'secondary'}


SECONDARY_METRIC_ORDER = {
    'oa': ('miou', 'aa', 'kappa'),
    'aa': ('miou', 'oa', 'kappa'),
    'miou': ('oa', 'aa', 'kappa'),
    'kappa': ('miou', 'oa', 'aa'),
}


def checkpoint_rank(metrics, metric_name, tie_break, epoch):
    if metric_name not in VALID_CHECKPOINT_METRICS:
        raise ValueError('Unsupported checkpoint metric: {}'.format(metric_name))
    if tie_break not in VALID_CHECKPOINT_TIE_BREAKS:
        raise ValueError('Unsupported checkpoint tie break: {}'.format(tie_break))

    missing = VALID_CHECKPOINT_METRICS.difference(metrics)
    if missing:
        raise ValueError('Missing validation metrics: {}'.format(sorted(missing)))

    primary = float(metrics[metric_name])
    if tie_break == 'latest':
        return primary, int(epoch)
    if tie_break == 'earliest':
        return primary, -int(epoch)

    secondary = tuple(
        float(metrics[name])
        for name in SECONDARY_METRIC_ORDER[metric_name]
    )
    return (primary,) + secondary + (-int(epoch),)


def should_replace_checkpoint(candidate_metrics, best_metrics, metric_name, tie_break,
                              candidate_epoch, best_epoch):
    if best_metrics is None:
        return True
    return checkpoint_rank(
        candidate_metrics,
        metric_name,
        tie_break,
        candidate_epoch,
    ) > checkpoint_rank(
        best_metrics,
        metric_name,
        tie_break,
        best_epoch,
    )
