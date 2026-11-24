"""Synthetic finite-state restoration and decision baselines."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

MODES = {0: "closed", 1: "pending_close", 2: "temporary_block", 3: "persistent_block"}

@dataclass(frozen=True)
class Config:
    case_id: str
    remote_mw: float = 12.0
    local_mw: float = 4.2
    local_execution_s: int = 15
    diagnostic_s: int = 10
    short_delay_s: int = 20
    long_delay_s: int = 60
    prior_age_s: int = 0
    repair_s: int = 120
    prior_weights: tuple = (20, 20, 20, 20, 20)
    horizon_s: int = 600
    query_s: int = 5
    cancel_confirmation_s: int = 5
    reset_s: int = 10
    close_s: int = 30
    acknowledgement_s: int = 5
    time_step_s: int = 5

def canonical(rows):
    merged = {}
    for mode, due, ack, weight in rows:
        if weight:
            merged[(mode, due, ack)] = merged.get((mode, due, ack), 0) + int(weight)
    divisor = 0
    for weight in merged.values():
        divisor = math.gcd(divisor, weight)
    return tuple((*key, weight // divisor) for key, weight in sorted(merged.items()))

class Model:
    def __init__(self, config):
        self.c = config
        self.counts = dict(transitions=0)

    def natural(self, world, start, end):
        mode,due,ack = world
        completion = start if mode == 0 else due if mode == 1 else math.inf
        if mode == 1 and due <= end:
            mode,due = 0,-1
        notification = ack >= 0 and start < ack <= end
        if notification:
            ack = -1
        return (mode,due,ack), max(0.,min(end,completion)-start), notification

    def transition(self, world, t, local_done, action):
        """Return (end, local_done, nextworld, observation, ENS in MWh).

        Macro retry and repair first confirm command termination. A delayed
        original closing can finish before that confirmation; in this branch
        the macro ends after the status/cancellation-confirmation interval.
        No follow-up close is issued before cancellation has been confirmed.
        """
        self.counts['transitions'] += 1
        c = self.c
        mode,due,ack = world
        if action == 'local_restore':
            duration = c.local_execution_s + c.acknowledgement_s
        elif action == 'poll': duration = c.query_s
        elif action == 'diagnose': duration = c.diagnostic_s
        elif action == 'wait': duration = c.time_step_s
        elif action == 'retry': duration = c.cancel_confirmation_s + c.reset_s + c.close_s + c.acknowledgement_s
        elif action == 'repair': duration = c.cancel_confirmation_s + c.repair_s + c.close_s + c.acknowledgement_s
        else: raise ValueError(action)
        macro = action in ('retry','repair')
        early_closed = mode == 0 or (mode == 1 and due <= t + c.cancel_confirmation_s)
        if macro and early_closed:
            duration = c.cancel_confirmation_s
        end = min(t+duration,c.horizon_s)
        dt = end-t
        nextworld, remote_unserved_time, notification = self.natural(world,t,end)
        response = action
        if macro:
            if early_closed:
                nextworld,remote_unserved_time,notification = self.natural(world,t,end)
                response = 'target_closed' if end == t+duration else 'horizon'
                if nextworld[0] == 0:
                    nextworld = (0,-1,-1)  # position is confirmed by the macro
            else:
                # Active command is definitively cancelled after the stated
                # confirmation time. Its later natural completion cannot occur.
                restore_allowed = action == 'repair' or mode in (1,2)
                physical_close = t+duration-c.acknowledgement_s if restore_allowed else math.inf
                remote_unserved_time = max(0.,min(end,physical_close)-t)
                if physical_close <= end:
                    nextworld = (0,-1,-1)
                elif mode == 1:
                    nextworld = (2,-1,-1)  # cancelled, electrically open, retry can close
                else:
                    nextworld = (mode,-1,-1)
                response = ('target_closed' if restore_allowed else 'refused') if end == t+duration else 'horizon'
                notification = False
        elif action == 'poll':
            response = {0:'closed',1:'pending',2:'open',3:'open'}[nextworld[0]]
            if nextworld[0] == 0: nextworld = (0,-1,-1)
        elif action == 'diagnose':
            response = {0:'closed',1:'pending',2:'temporary_block',3:'persistent_block'}[nextworld[0]]
            if nextworld[0] == 0: nextworld = (0,-1,-1)
        elif action == 'local_restore':
            response = 'local_complete' if t+c.local_execution_s <= end else 'horizon'
        if local_done:
            local_unserved_time = 0.
        elif action == 'local_restore':
            local_unserved_time = min(dt,c.local_execution_s)
        else:
            local_unserved_time = dt
        next_local = local_done or (action == 'local_restore' and t+c.local_execution_s <= end)
        observation = (end,response,'target_ack' if notification else 'no_target_ack')
        energy = (c.local_mw*local_unserved_time+c.remote_mw*remote_unserved_time)/3600.
        return end,next_local,nextworld,observation,energy

    @lru_cache(None)
    def outcomes(self, belief, t, local_done, action):
        groups = {}
        total = sum(w[3] for w in belief)
        immediate = 0.
        for mode,due,ack,weight in belief:
            end,nlocal,nworld,obs,cost = self.transition((mode,due,ack),t,local_done,action)
            immediate += weight*cost/total
            key = (end,nlocal,obs)
            groups.setdefault(key,[]).append((*nworld,weight))
        results = []
        for (end,nlocal,obs),rows in sorted(groups.items()):
            prob = sum(r[3] for r in rows)/total
            results.append((prob,canonical(rows),end,nlocal,obs))
        return immediate,tuple(results)

    def stop_cost(self, belief,t,local_done):
        c = self.c
        remain = c.horizon_s-t
        local = 0. if local_done else c.local_mw*remain/3600.
        total = sum(w[3] for w in belief)
        remote = 0.
        for mode,due,ack,weight in belief:
            unserved = 0 if mode==0 else min(remain,max(0,due-t)) if mode==1 else remain
            remote += weight/total*c.remote_mw*unserved/3600.
        return local+remote

    def actions(self,belief,t,local_done):
        if t >= self.c.horizon_s: return ()
        result = []
        if not local_done: result.append('local_restore')
        if any(w[0] != 0 for w in belief):
            result += ['retry','repair']
        # Only genuinely informative direct inspections are retained. Pure
        # passage of time is already available through the common wait action.
        for action in ('poll','diagnose'):
            _,outcomes = self.outcomes(belief,t,local_done,action)
            if len(outcomes)>1: result.append(action)
        if any(w[0]==1 or w[2]>t for w in belief): result.append('wait')
        return tuple(result)

    @lru_cache(None)
    def optimal(self,belief,t,local_done):
        best = (self.stop_cost(belief,t,local_done),'stop')
        for action in self.actions(belief,t,local_done):
            immediate,outcomes = self.outcomes(belief,t,local_done,action)
            cost = immediate+sum(p*self.optimal(b,end,ld)[0] for p,b,end,ld,_ in outcomes)
            if cost < best[0]-1e-12: best=(cost,action)
        return best

    @lru_cache(None)
    def ce_action(self,belief,t,local_done):
        # Certainty-equivalent plan uses the most likely current world, then
        # is recomputed after each real observation. No future outcome is given.
        world=max(belief,key=lambda x:(x[3],-x[0]))
        single=((*world[:3],1),)
        action=self.optimal(single,t,local_done)[1]
        if action=='stop' and any(w[0]!=0 for w in belief) and t<self.c.horizon_s:
            # A strengthened CE controller verifies apparent completion rather
            # than simply abandoning a possibly still unserved target because
            # its MAP hypothesis happened to be "already closed".
            available=self.actions(belief,t,local_done)
            if 'poll' in available:return 'poll'
            if 'diagnose' in available:return 'diagnose'
            if 'wait' in available:return 'wait'
        return action

    @lru_cache(None)
    def ce_value(self,belief,t,local_done):
        action=self.ce_action(belief,t,local_done)
        if action=='stop': return self.stop_cost(belief,t,local_done)
        immediate,outs=self.outcomes(belief,t,local_done,action)
        return immediate+sum(p*self.ce_value(b,end,ld) for p,b,end,ld,_ in outs)

    @lru_cache(None)
    def rollout_value(self,belief,t,local_done,depth):
        if depth==0:return self.ce_value(belief,t,local_done)
        best=self.stop_cost(belief,t,local_done)
        for action in self.actions(belief,t,local_done):
            immediate,outs=self.outcomes(belief,t,local_done,action)
            value=immediate+sum(p*self.rollout_value(b,end,ld,depth-1) for p,b,end,ld,_ in outs)
            best=min(best,value)
        return best

    @lru_cache(None)
    def policy_action(self,policy,belief,t,local_done):
        if policy=='optimal_belief_dp': return self.optimal(belief,t,local_done)[1]
        if policy=='certainty_equivalent_mpc': return self.ce_action(belief,t,local_done)
        available=self.actions(belief,t,local_done)
        if policy=='entropy_per_second':
            rates=[]
            for a in available:
                if a not in ('poll','diagnose'):continue
                _,outs=self.outcomes(belief,t,local_done,a)
                entropy=-sum(p*math.log2(p) for p,*_ in outs)
                duration=sum(p*(end-t) for p,b,end,ld,_ in outs)
                rates.append((entropy/duration,a))
            if rates:return max(rates)[1]
            return self.policy_action('qmdp',belief,t,local_done)
        best=(self.stop_cost(belief,t,local_done),'stop')
        for action in available:
            immediate,outs=self.outcomes(belief,t,local_done,action)
            if policy=='qmdp':
                # Optimistic full-observability tail, standard QMDP heuristic.
                tail=0.
                for p,b,end,ld,_ in outs:
                    total=sum(w[3] for w in b)
                    tail+=p*sum(w[3]/total*self.optimal(((*w[:3],1),),end,ld)[0] for w in b)
            elif policy=='one_step_voi_rollout':
                # A generic value-of-information rollout with an adaptive CE
                # continuation. This is an established baseline, not our novelty.
                tail=sum(p*self.ce_value(b,end,ld) for p,b,end,ld,_ in outs)
            elif policy=='two_step_voi_rollout':
                tail=sum(p*self.rollout_value(b,end,ld,1) for p,b,end,ld,_ in outs)
            else: raise ValueError(policy)
            val=immediate+tail
            if val<best[0]-1e-12:best=(val,action)
        return best[1]

    @lru_cache(None)
    def evaluate(self,policy,belief,t,local_done):
        action=self.policy_action(policy,belief,t,local_done)
        if action=='stop':
            return (self.stop_cost(belief,t,local_done),0.,0.,0.,0.)
        immediate,outs=self.outcomes(belief,t,local_done,action)
        # ENS, direct query count, retries, repairs, elapsed decision time.
        value=[immediate,float(action in ('poll','diagnose')),float(action=='retry'),float(action=='repair'),sum(p*(end-t) for p,b,end,ld,_ in outs)]
        for p,b,end,ld,_ in outs:
            tail=self.evaluate(policy,b,end,ld)
            for j in range(5):value[j]+=p*tail[j]
        return tuple(value)

    def trajectory(self,policy,initial_world,belief):
        world=initial_world
        t=0
        local_done=False
        energy=0.
        trace=[]
        while t<self.c.horizon_s:
            action=self.policy_action(policy,belief,t,local_done)
            if action=='stop':
                residual=self.stop_cost(((*world,1),),t,local_done)
                energy+=residual
                trace.append(dict(time=t,action='stop',tail_ens_mwh=residual))
                break
            end,nlocal,nworld,observation,cost=self.transition(world,t,local_done,action)
            _,outs=self.outcomes(belief,t,local_done,action)
            matches=[b for p,b,e,ld,obs in outs if e==end and ld==nlocal and obs==observation]
            assert len(matches)==1
            trace.append(dict(time=t,action=action,end=end,observation=observation,
                              target_mode_before=world[0],target_mode_after=nworld[0],ens_mwh=cost))
            energy+=cost
            world,belief,t,local_done=nworld,matches[0],end,nlocal
        return energy,trace


def clear(m):
    for n in ('outcomes','optimal','ce_action','ce_value','rollout_value','policy_action','evaluate'):
        getattr(m,n).cache_clear()

def state_view(m,b,t,ld):
    c=asdict(m.c)
    for k in ('case_id','prior_weights','prior_age_s'):c.pop(k)
    total=sum(w[3] for w in b)
    return dict(current_time_s=t,local_restored=ld,parameters=c,posterior=[dict(mode=MODES[x],physical_due_s=d,ack_due_s=a,probability=w/total) for x,d,a,w in b],available_actions=list(m.actions(b,t,ld))+['stop'])
