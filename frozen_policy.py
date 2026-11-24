"""Fixed arithmetic rule generated offline and frozen before evaluation."""

def choose_action(state):
    p = state['parameters']
    posterior = state['posterior']
    available = state['available_actions']
    now = state['current_time_s']
    h = max(0.0, p.get('horizon_s', 600.0) - now)
    r = p['remote_mw']
    l = p['local_mw']
    if state['local_restored']:
        l = 0.0
    e = p['local_execution_s']
    ack = p['acknowledgement_s']
    ld = e + ack
    c = p['cancel_confirmation_s']
    rt = c + p['reset_s'] + p['close_s']
    rp = c + p['repair_s'] + p['close_s']
    rd = rt + ack
    pd = rp + ack
    query = p['query_s']
    diagnostic = p['diagnostic_s']
    step = max(0.000000001, p['time_step_s'])
    total = 0.0
    for w in posterior:
        total += w['probability']
    if total <= 0.0 or h <= 0.0:
        for action in available:
            if action == 'stop':
                return action
        return available[0]
    worlds = []
    stop_score = l * h
    for w in posterior:
        q = w['probability'] / total
        mode = w['mode']
        due = h + rt + rp + ld
        if mode == 'closed':
            due = 0.0
        elif mode == 'pending_close':
            physical = w.get('physical_due_s', -1)
            if physical >= 0.0:
                due = max(0.0, physical - now)
        worlds.append((q, mode, due))
        stop_score += q * r * min(h, due)
    scores = {'stop': stop_score}
    local_score = stop_score
    for placement in range(2):
        if placement == 1 and l == 0.0:
            continue
        shift = 0.0
        weight = l
        local_base = 0.0
        if placement == 1:
            shift = min(h, ld)
            weight = 0.0
            local_base = l * min(h, e)
        remaining = max(0.0, h - shift)
        local_delay = 0.0
        if weight > 0.0:
            local_delay = ld
        retry_score = local_base
        repair_score = local_base
        natural_score = local_base + weight * min(remaining, e)
        for w in worlds:
            q = w[0]
            mode = w[1]
            due = w[2]
            before = r * min(shift, due)
            delay = max(0.0, due - shift)
            natural_score += q * r * min(h, due)
            if mode == 'closed' or (mode == 'pending_close' and delay <= c):
                rc = r * min(remaining, delay) + weight * min(remaining, c + e)
                pc = rc
            else:
                pc = r * min(remaining, rp) + weight * min(remaining, pd + e)
                if mode == 'persistent_block':
                    first_local = r * min(remaining, rd + local_delay + rp) + weight * min(remaining, rd + e)
                    first_repair = r * min(remaining, rd + rp) + weight * min(remaining, rd + pd + e)
                    abandon = r * remaining + weight * min(remaining, rd + e)
                    rc = min(first_local, first_repair, abandon)
                else:
                    rc = r * min(remaining, rt) + weight * min(remaining, rd + e)
            retry_score += q * (before + rc)
            repair_score += q * (before + pc)
        if placement == 0:
            scores['retry'] = retry_score
            scores['repair'] = repair_score
        else:
            local_score = min(retry_score, repair_score, natural_score)
        for information in range(2):
            duration = query
            name = 'poll'
            if information == 1:
                duration = diagnostic
                name = 'diagnose'
            elapsed = min(remaining, duration)
            tail = max(0.0, remaining - elapsed)
            origin = shift + elapsed
            prefix = local_base + weight * elapsed
            closed_cost = 0.0
            pending_natural = 0.0
            pending_retry = 0.0
            pending_local_retry = 0.0
            pending_repair = 0.0
            pending_local_repair = 0.0
            temporary = [0.0, 0.0, 0.0, 0.0, 0.0]
            persistent = [0.0, 0.0, 0.0, 0.0, 0.0]
            temporary_mass = 0.0
            persistent_mass = 0.0
            for w in worlds:
                q = w[0]
                mode = w[1]
                due = w[2]
                prefix += q * r * min(h, origin, due)
                delay = max(0.0, due - origin)
                if mode == 'closed' or (mode == 'pending_close' and delay <= 0.0):
                    closed_cost += q * weight * min(tail, e)
                elif mode == 'pending_close':
                    pending_natural += q * (r * min(tail, delay) + weight * min(tail, e))
                    tr = rt
                    tp = rp
                    dr = rd
                    dp = pd
                    if delay <= c:
                        tr = delay
                        tp = delay
                        dr = c
                        dp = c
                    pending_retry += q * (r * min(tail, tr) + weight * min(tail, dr + e))
                    pending_repair += q * (r * min(tail, tp) + weight * min(tail, dp + e))
                    lr = local_delay + rt
                    lp = local_delay + rp
                    if delay <= local_delay + c:
                        lr = delay
                        lp = delay
                    pending_local_retry += q * (r * min(tail, lr) + weight * min(tail, e))
                    pending_local_repair += q * (r * min(tail, lp) + weight * min(tail, e))
                elif mode == 'temporary_block':
                    temporary_mass += q
                    temporary[0] += q * (r * min(tail, rt) + weight * min(tail, rd + e))
                    temporary[1] += q * (r * min(tail, local_delay + rt) + weight * min(tail, e))
                    temporary[2] += q * (r * min(tail, rp) + weight * min(tail, pd + e))
                    temporary[3] += q * (r * min(tail, local_delay + rp) + weight * min(tail, e))
                    temporary[4] += q * (r * tail + weight * min(tail, e))
                else:
                    persistent_mass += q
                    fallback_local = r * min(tail, rd + local_delay + rp) + weight * min(tail, rd + e)
                    fallback_repair = r * min(tail, rd + rp) + weight * min(tail, rd + pd + e)
                    fallback_stop = r * tail + weight * min(tail, rd + e)
                    persistent[0] += q * min(fallback_local, fallback_repair, fallback_stop)
                    persistent[1] += q * (r * min(tail, local_delay + rd + rp) + weight * min(tail, e))
                    persistent[2] += q * (r * min(tail, rp) + weight * min(tail, pd + e))
                    persistent[3] += q * (r * min(tail, local_delay + rp) + weight * min(tail, e))
                    persistent[4] += q * (r * tail + weight * min(tail, e))
            pending_cost = min(pending_natural, pending_retry, pending_local_retry, pending_repair, pending_local_repair)
            # A pending contact can be given until a physical deadline's
            # cancellation window, then checked by retry or repair itself.
            # Evaluate shared deadlines, never a separate choice per world.
            for target in worlds:
                if target[1] != 'pending_close' or target[2] <= origin or target[2] >= h:
                    continue
                for order in range(2):
                    if order == 1 and weight == 0.0:
                        continue
                    base = 0.0
                    if order == 1:
                        base = local_delay
                    slack = max(0.0, target[2] - origin - c - base)
                    ticks = int(slack / step)
                    if ticks * step < slack - 0.000000001:
                        ticks += 1
                    start = base + ticks * step
                    if start >= tail:
                        continue
                    delayed_retry = 0.0
                    delayed_repair = 0.0
                    for w in worlds:
                        if w[1] != 'pending_close' or w[2] <= origin:
                            continue
                        q = w[0]
                        delay = w[2] - origin
                        tr = start + rt
                        tp = start + rp
                        dr = start + rd
                        dp = start + pd
                        if delay <= start + c:
                            tr = delay
                            tp = delay
                            dr = start + c
                            dp = start + c
                        lr = dr + e
                        lp = dp + e
                        if order == 1:
                            lr = e
                            lp = e
                        delayed_retry += q * (r * min(tail, tr) + weight * min(tail, lr))
                        delayed_repair += q * (r * min(tail, tp) + weight * min(tail, lp))
                    pending_cost = min(pending_cost, delayed_retry, delayed_repair)
            if information == 1:
                block_cost = min(temporary) + min(persistent[2], persistent[3], persistent[4])
            else:
                block_cost = temporary[4] + persistent[4]
                for option in range(4):
                    block_cost = min(block_cost, temporary[option] + persistent[option])
                # An open poll leaves only the two blocked causes. Diagnosis
                # can then avoid a failed retry without charging its duration
                # to already closed or naturally completing branches.
                if temporary_mass > 0.0 and persistent_mass > 0.0:
                    d = diagnostic
                    temp_retry = r * min(tail, d + rt) + weight * min(tail, d + rd + e)
                    temp_local = r * min(tail, d + local_delay + rt) + weight * min(tail, d + e)
                    temp_repair = r * min(tail, d + rp) + weight * min(tail, d + pd + e)
                    temp_local_repair = r * min(tail, d + local_delay + rp) + weight * min(tail, d + e)
                    abandon = r * tail + weight * min(tail, d + e)
                    pers_repair = r * min(tail, d + rp) + weight * min(tail, d + pd + e)
                    pers_local = r * min(tail, d + local_delay + rp) + weight * min(tail, d + e)
                    diagnosed = temporary_mass * min(temp_retry, temp_local, temp_repair, temp_local_repair, abandon)
                    diagnosed += persistent_mass * min(pers_repair, pers_local, abandon)
                    block_cost = min(block_cost, diagnosed)
                    if weight > 0.0:
                        temp_after_local = r * min(tail, ld + d + min(rt, rp)) + weight * min(tail, e)
                        pers_after_local = r * min(tail, ld + d + rp) + weight * min(tail, e)
                        local_abandon = r * tail + weight * min(tail, e)
                        diagnosed_local = temporary_mass * min(temp_after_local, local_abandon)
                        diagnosed_local += persistent_mass * min(pers_after_local, local_abandon)
                        block_cost = min(block_cost, diagnosed_local)
            information_score = prefix + closed_cost + pending_cost + block_cost
            if placement == 0:
                scores[name] = information_score
            else:
                local_score = min(local_score, information_score)
    if l > 0.0:
        scores['local_restore'] = local_score
    # Waiting is useful even in a mixed posterior when a near physical
    # completion can be caught by the subsequent termination check.
    # These are closed-form shared-deadline scheduling scores.
    waiting_score = stop_score
    for target in worlds:
        if target[1] != 'pending_close' or target[2] >= h:
            continue
        slack = max(step, target[2] - c)
        ticks = int(slack / step)
        if ticks * step < slack - 0.000000001:
            ticks += 1
        start = ticks * step
        if start >= h:
            continue
        tail = h - start
        delayed_retry = l * start
        delayed_repair = l * start
        local_delay = 0.0
        if l > 0.0:
            local_delay = ld
        for w in worlds:
            q = w[0]
            mode = w[1]
            due = w[2]
            before = r * min(start, due)
            delay = max(0.0, due - start)
            if mode == 'closed' or (mode == 'pending_close' and delay <= c):
                rc = r * min(tail, delay) + l * min(tail, c + e)
                pc = rc
            else:
                pc = r * min(tail, rp) + l * min(tail, pd + e)
                if mode == 'persistent_block':
                    fallback_local = r * min(tail, rd + local_delay + rp) + l * min(tail, rd + e)
                    fallback_repair = r * min(tail, rd + rp) + l * min(tail, rd + pd + e)
                    fallback_stop = r * tail + l * min(tail, rd + e)
                    rc = min(fallback_local, fallback_repair, fallback_stop)
                else:
                    rc = r * min(tail, rt) + l * min(tail, rd + e)
            delayed_retry += q * (before + rc)
            delayed_repair += q * (before + pc)
        waiting_score = min(waiting_score, delayed_retry, delayed_repair)
    scores['wait'] = waiting_score
    best = available[0]
    best_score = scores.get(best, stop_score + 1.0)
    for action in available:
        score = scores.get(action, stop_score + 1.0)
        if score < best_score - 0.000000001:
            best_score = score
            best = action
        elif abs(score - best_score) <= 0.000000001 and action == 'stop':
            best = action
    return best
