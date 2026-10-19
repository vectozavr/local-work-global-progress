"""The team protocol (Algorithms 1 and 2 of the paper).

Each round, every agent searches its own block in parallel for up to H answers against the frozen plan, scoring its
candidates with its local loss plus the correction for the collisions with other agents' robots that the coordinator
already knows about. The coordinator then verifies the proposals on the full plan and merges them:
merge='best_single' accepts the best improving block (Algorithm 1), 'best_subset' the checked combination of changed
blocks with the lowest loss (Algorithm 2), and 'all' every changed block unchecked. no_correction=True withholds the
correction and the global collision feedback. The run resumes from saved answers after an interruption.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from itertools import combinations, combinations as _combinations
import json
import threading
import time

from .feedback import describe_attempt
from .files import read, rows, save
from .llm import OpenRouterError, parse_program_response, recovered_answer
from .prompt_team import build_team_prompt
from .reference import joint_block_minimum, minimum_score
from .world import World, digest

PROMPT_VERSION = 'team-v2'

SCHEDULER = 'independent-private-searches-v1-team-prompt-v2'


def available(w, b, programs, known):
    owned = set(w.groups[b])
    need = owned | {j if i in owned else i for i,j in known if (i in owned)!=(j in owned)}
    traces = {i:w.trace(i,programs[i]) for i in need}
    events = []
    for i,j in combinations(range(len(w.ids)),2):
        if {i,j} <= owned or ((i,j) in known and (i in owned)!=(j in owned)):
            events += w.collisions(traces[i].positions,traces[j].positions,i,j)
    return sum(traces[i].cost for i in owned)+w.M*len(events), events


def certificate(w, b, base, known):
    group = w.groups[b]
    if len(group)==1:
        i=group[0]
        r=minimum_score(w,base,i,[j if i==k else k for k,j in known if i in (k,j)])
        return dict(kind='exact_single_robot_minimum',lower_bound=r['score'],states_expanded=r['states_expanded'])
    r=joint_block_minimum(w,group,base,known,seconds=5.)
    if r['minimum_certified']:
        return dict(kind='exact_joint_minimum_CBS_with_penalty_bounds',lower_bound=r['exact_minimum'],
                    states_expanded=r['hard_routing']['expanded'])
    unary=[minimum_score(w,base,i,[]) for i in group]
    return dict(kind='sum_of_unary_lower_bounds',lower_bound=sum(r['score'] for r in unary),
                states_expanded=sum(r['states_expanded'] for r in unary))


def run(args, spec, model, root):
    w=World(spec);path=root/f'seed-{args.seed}-learned';protocol=path/'protocol'
    protocol.mkdir(exist_ok=True);(protocol/'attempts').mkdir(exist_ok=True)
    statefile=root/'STATE.json'
    if not statefile.exists():
        save(protocol/'instance.json',spec)
        save(statefile,dict(state='RUNNING',started_unix=time.time(),epoch=0,base=[[] for _ in w.ids],
            known=[],last_invalid=[None]*len(w.groups),global_feedback=[None]*len(w.groups),history=[],
            stopping_check_seconds=0.,verification_seconds=0.,resumes=0))
    state=read(statefile)
    if state['state']=='COMPLETED':return read(protocol/'result.json')
    patience=getattr(args,'no_improvement_patience',5)
    if patience is not None and patience < 1:
        raise ValueError('No-improvement patience must be positive or None')
    merge=getattr(args,'merge','best_single')
    if merge not in ('best_single','best_subset','all'):raise ValueError(f'Unknown merge rule {merge}')
    no_correction=getattr(args,'no_correction',False)
    if no_correction:
        if merge!='best_single':raise ValueError('The no-correction ablation was run only with --merge best_single')
        assert not state['known'], 'Ablation cannot inherit known external pairs'
        assert all(v is None for v in state['global_feedback']), 'Ablation cannot inherit global feedback'
    nonimproving=0
    for previous in reversed(state['history']):
        if previous.get('improved',bool(previous['accepted'])):break
        nonimproving+=1
    if args.resume:
        state['resumes']+=1;save(statefile,state)
    lock=threading.Lock()

    def all_calls():
        return sorted([read(p) for p in (protocol/'attempts').glob('*.json')],
                      key=lambda c:(c['epoch'],c['block'],c['step']))

    def archive_call(file, record):
        with lock:
            save(file,record)
            temporary=protocol/'calls.jsonl.tmp'
            temporary.write_text(''.join(json.dumps(c)+'\n' for c in all_calls()))
            temporary.replace(protocol/'calls.jsonl')

    while state['epoch']<args.epochs:
        epoch=state['epoch'];frozen=copy.deepcopy(state['base']);known=set(map(tuple,state['known']))
        current=w.evaluate_programs(frozen);before=current['F']
        # A crash after committing success but before writing result.json must
        # finalize that same round, without appending an empty extra round.
        target=getattr(args,'target_F',None)
        if (state['history'] and current['delivered']==len(w.ids) and not current['collisions']
                and (target is None or current['F']<=target)):
            break
        if patience is not None and nonimproving>=patience:
            break
        roundfile=protocol/f'round-{epoch:02d}-state.json'
        if roundfile.exists():roundstate=read(roundfile)
        else:
            tic=time.perf_counter();certs=[certificate(w,b,frozen,known) for b in range(len(w.groups))]
            prep=time.perf_counter()-tic
            roundstate=dict(started_unix=time.time()-prep,certificates=certs,stopping_check_seconds=prep,
                            search_started_unix=time.time())
            save(roundfile,roundstate)

        def search(b):
            blockfile=protocol/f'round-{epoch:02d}-block-{b}.json'
            if blockfile.exists():block=read(blockfile)
            else:
                baseline,_=available(w,b,frozen,known)
                lower=roundstate['certificates'][b]['lower_bound']
                assert baseline>=lower-1e-9
                block=dict(started_unix=time.time(),step=0,attempts=0,proposal=copy.deepcopy(frozen),best=0.,log=[],
                    previous=copy.deepcopy(state['last_invalid'][b]),last_invalid=copy.deepcopy(state['last_invalid'][b]),
                    stopped=abs(baseline-lower)<=1e-9,baseline_value=baseline,finished_unix=None)
                save(blockfile,block)
            while block['step']<args.H and not block['stopped']:
                step=block['step'];context=dict(epoch=epoch,block=b,step=step)
                callfile=protocol/'attempts'/f'e{epoch:02d}-b{b}-s{step}.json'
                value0,events=available(w,b,frozen,known)
                prompt=build_team_prompt(w,b,frozen,known,value=value0,events=events,log=block['log'],
                                         global_feedback=state['global_feedback'][b],epoch=epoch,step=step,H=args.H,merge=merge,
                                         correction=not no_correction)
                seed=args.seed+epoch*1000+step*100+b
                if callfile.exists():record=read(callfile)
                else:
                    journal=rows(path/'api-requests.jsonl')
                    saved=[r for r in journal if r.get('call_context')==context and not r.get('error')]
                    if saved:
                        assert saved[-1]['request']==model.payload(prompt,seed)
                        answer=recovered_answer(saved[-1],seed)
                    else:
                        failures=sum(r.get('call_context')==context and bool(r.get('error')) for r in journal)
                        while True:
                            if failures>=getattr(args,'transport_attempts',3):
                                raise OpenRouterError('Transport attempt budget exhausted; inspect the journal before resuming')
                            try:
                                print(json.dumps(dict(event='request_started',**context,seconds=time.time()-state['started_unix'])),flush=True)
                                answer=model.one(prompt,seed,call_context=context);break
                            except OpenRouterError:
                                if getattr(args,'format_attempts',0):raise
                                failures+=1
                                if failures>=getattr(args,'transport_attempts',3):raise
                                time.sleep(30.)
                    record=dict(**context,prompt=prompt,**answer,elapsed_seconds=time.time()-state['started_unix'])
                    archive_call(callfile,record)
                d=None
                try:
                    d=parse_program_response(record['text']);owned={w.ids[i] for i in w.groups[b]}
                    if set(d)!=owned:raise ValueError(f'Expected exactly {sorted(owned)}; missing {sorted(owned-set(d))}, extra {sorted(set(d)-owned)}')
                    candidate=copy.deepcopy(frozen);errors=[]
                    for i in w.groups[b]:
                        try:w.trace(i,d[w.ids[i]]);candidate[i]=d[w.ids[i]]
                        except (ValueError,TypeError) as exc:errors.append(f'{w.ids[i]}: {exc}')
                    if errors:raise ValueError('; '.join(errors))
                    value,events=available(w,b,candidate,known);score=value-block['baseline_value']
                    _,base_events=available(w,b,frozen,known);own=set(w.groups[b])
                    ext=lambda es:sum(not set(e['pair'])<=own for e in es)
                    correction=w.M*(ext(events)-ext(base_events))
                    block['previous']=dict(programs=d,score_change=score,collisions=events,
                        local_cost=value-w.M*ext(events),local_score_change=score-correction,
                        correction_score_change=correction,
                        feedback='Valid program. Try to reduce arrival time or known collision penalties; the best verified program is retained.',
                        program_views=describe_attempt(w,w.groups[b],d,frozen,known))
                    block['log'].append(dict(step=step,programs=d,valid=True,value=value,events=events))
                    if score<block['best']:block['best']=score;block['proposal']=candidate
                    block['last_invalid']=None
                    record.update(valid=True,score=score,candidate=candidate,
                                  local_score_change=score-correction,correction_score_change=correction)
                    assert value>=roundstate['certificates'][b]['lower_bound']-1e-9
                    if abs(value-roundstate['certificates'][b]['lower_bound'])<=1e-9:
                        block['stopped']=True;record['stop_reason']='certified_local_minimum'
                except (ValueError,TypeError,KeyError) as exc:
                    record.update(valid=False,error=str(exc))
                    block['previous']=dict(error=str(exc),attempted_response=record['text'],epoch=epoch)
                    if isinstance(d,dict):
                        block['previous'].update(programs=d,program_views=describe_attempt(w,w.groups[b],d,frozen,known))
                    block['last_invalid']=copy.deepcopy(block['previous'])
                    block['log'].append(dict(step=step,programs=d if isinstance(d,dict) else {},valid=False,error=str(exc)))
                archive_call(callfile,record)
                block['step']+=1;block['attempts']+=1;save(blockfile,block)
                print(json.dumps(dict(event='attempt_checked',**context,valid=record['valid'],
                    error=record.get('error'),seconds=record['elapsed_seconds'])),flush=True)
            if block['finished_unix'] is None:
                block['finished_unix']=time.time();save(blockfile,block)
            block['wall_seconds']=block['finished_unix']-block['started_unix']
            block['api_seconds']=sum(c['seconds'] for c in all_calls() if c['epoch']==epoch and c['block']==b)
            return block

        with ThreadPoolExecutor(max_workers=len(w.groups)) as pool:
            blocks=list(pool.map(search,range(len(w.groups))))
        search_finished=time.time();proposals=[b['proposal'] for b in blocks]
        tic=time.perf_counter();checked=[w.evaluate_programs(p) for p in proposals]
        gains=[max(0,before-v['F']) for v in checked]
        selected=max(range(len(gains)),key=lambda b:(gains[b],-b));accepted=[]
        base=copy.deepcopy(frozen);combos=[]
        if merge=='best_single':
            if gains[selected]>0:base=proposals[selected];accepted=[selected]
        else:
            # Blocks whose returned proposal differs from the frozen base; every combination of them is checked.
            changed=[b for b in range(len(w.groups)) if any(proposals[b][i]!=frozen[i] for i in w.groups[b])]
            def merged(S):
                x=copy.deepcopy(frozen)
                for b in S:
                    for i in w.groups[b]:x[i]=copy.deepcopy(proposals[b][i])
                return x
            for r in range(1,len(changed)+1):
                for S in _combinations(changed,r):
                    v=w.evaluate_programs(merged(S));combos.append(dict(blocks=list(S),F=v['F'],collisions=v['collisions'],events=v['events']))
            if merge=='best_subset':
                good=[c for c in combos if c['F']<before-1e-9]
                if good:
                    best=min(good,key=lambda c:(c['F'],len(c['blocks']),c['blocks']));base=merged(best['blocks']);accepted=list(best['blocks'])
            elif changed:   # merge=='all': install every changed block together, whatever the loss
                base=merged(changed);accepted=list(changed)
        current=w.evaluate_programs(base)
        assert merge=='all' or current['F']<=before+1e-9
        best_before=state.get('best_F',w.evaluate_programs([[] for _ in w.ids])['F'] if not state['history'] else min(h['F'] for h in state['history']))
        improved=current['F']<best_before-1e-9
        verification=time.perf_counter()-tic
        if no_correction:
            # Global events remain in the journal; none enter any local score or prompt.
            assert not known
            feedback=[None]*len(w.groups)
        else:
            feedback=[]
            for c in combos:
                for e in c['events']:
                    i,j=e['pair']
                    if w.owner[i]!=w.owner[j]:known.add((i,j))
            for b,v in enumerate(checked):
                for e in v['events']:
                    i,j=e['pair']
                    if w.owner[i]!=w.owner[j]:known.add((i,j))
                conflicts=[e for e in v['events'] if set(e['pair']) & set(w.groups[b])]
                feedback.append(None if not conflicts else dict(global_round=epoch+1,
                    programs={w.ids[i]:proposals[b][i] for i in w.groups[b]},
                    collisions=[{**e,'pair':[w.ids[i] for i in e['pair']]} for e in conflicts],
                    message='This returned program collided during global verification at the listed ticks. Use the refreshed peer positions to revise it.'))
        elapsed=time.time()-state['started_unix']
        row=dict(epoch=epoch,base=frozen,proposals=proposals,next=base,F=current['F'],accepted=accepted,merge=merge,improved=improved,
            best_F=min(best_before,current['F']),combinations=[{k:c[k] for k in ('blocks','F','collisions')} for c in combos],
            delivered=current['delivered'],collisions=current['collisions'],known_before=state['known'],known_after=sorted(known),
            seconds=elapsed,verified_F=[v['F'] for v in checked],proposal_events=[v['events'] for v in checked],gains=gains,
            local_attempts=[b['attempts'] for b in blocks],
            local_stop_reasons=['certified_local_minimum' if b['stopped'] else 'budget_exhausted' for b in blocks],
            stopping_certificates=roundstate['certificates'],
            timing=dict(agent_wall_seconds=[b['wall_seconds'] for b in blocks],agent_api_seconds=[b['api_seconds'] for b in blocks],
                slowest_agent_seconds=max(b['wall_seconds'] for b in blocks),
                parallel_search_seconds=search_finished-roundstate['search_started_unix'],
                round_wall_seconds=elapsed-(state['history'][-1]['seconds'] if state['history'] else 0),
                verification_seconds=verification,stopping_check_seconds=roundstate['stopping_check_seconds']))
        state['history'].append(row);state.update(base=base,known=sorted(known),epoch=epoch+1,
            last_invalid=[b['last_invalid'] for b in blocks],global_feedback=feedback)
        if improved or 'best_base' not in state:
            state['best_F']=min(best_before,current['F']);state['best_base']=copy.deepcopy(base) if improved else copy.deepcopy(state.get('best_base',[[] for _ in w.ids]))
        nonimproving=0 if improved else nonimproving+1
        state['stopping_check_seconds']+=roundstate['stopping_check_seconds'];state['verification_seconds']+=verification
        save(statefile,state)
        (protocol/'epochs.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in state['history']))
        print(json.dumps(dict(event='global_round',round=epoch+1,F=current['F'],delivered=current['delivered'],
            collisions=current['collisions'],accepted=accepted,local_attempts=row['local_attempts'],
            known_pairs=len(known),seconds=elapsed,timing=row['timing'])),flush=True)
        if (current['delivered']==len(w.ids) and current['collisions']==0
                and (target is None or current['F']<=target)):break
    calls=all_calls();costs=[c.get('cost_usd') for c in calls]
    reasoning=[c.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens') for c in calls]
    last_installed=w.evaluate_programs(state['base'])
    best_programs=state.get('best_base',state['base']) if merge=='all' else state['base']
    if merge=='all' and w.evaluate_programs(state['base'])['F']<=w.evaluate_programs(best_programs)['F']:best_programs=state['base']
    current=w.evaluate_programs(best_programs);success=current['delivered']==len(w.ids) and not current['collisions']
    target=getattr(args,'target_F',None)
    reached=bool(success and (target is None or current['F']<=target))
    journal=rows(path/'api-requests.jsonl')
    http_costs=[r.get('response',{}).get('usage',{}).get('cost') for r in journal]
    safe_rounds=[h for h in state['history'] if h['delivered']==len(w.ids) and not h['collisions']]
    target_rounds=[h for h in safe_rounds if target is None or h['F']<=target]
    result=dict(domain='free',prompt_version=PROMPT_VERSION,scheduler=SCHEDULER,mode='local_only' if no_correction else 'learned',
        cross_block_correction=not no_correction,global_collision_feedback=not no_correction,solver='http',seed=args.seed,
        H=args.H,epochs=state['epoch'],epoch_budget=args.epochs,F=current['F'],F_star=None,
        assumption3='Not certified for arbitrary command arrays.',spec_sha256=digest(spec),
        delivered=current['delivered'],collisions=current['collisions'],first_success_seconds=safe_rounds[0]['seconds'] if safe_rounds else None,
        target_F=target,first_target_seconds=target_rounds[0]['seconds'] if target is not None and target_rounds else None,
        reached_target=reached,wall_seconds=time.time()-state['started_unix'],
        final_programs=dict(zip(w.ids,best_programs)),history=state['history'],merge=merge,last_installed_F=last_installed['F'],
        last_installed_programs=dict(zip(w.ids,state['base'])),model_calls=len(calls),solver_calls=len(calls),
        invalid_responses=sum(not c['valid'] for c in calls),input_tokens=sum(c['usage']['prompt_tokens'] for c in calls),
        output_tokens=sum(c['usage']['completion_tokens'] for c in calls),
        llm_batch_seconds=sum(h['timing']['parallel_search_seconds'] for h in state['history']),
        parallel_critical_path_seconds=sum(h['timing']['slowest_agent_seconds'] for h in state['history']),
        summed_agent_api_seconds=sum(c['seconds'] for c in calls),verification_seconds=state['verification_seconds'],
        certified_stop=True,stopping_check_seconds=state['stopping_check_seconds'],
        singleton_verifications=state['epoch']*len(w.groups),joint_replay_checks=state['epoch'],model=model.metadata,
        run_status='COMPLETED',completed_protocol=True,solved=bool(success),
        termination=(('target_reached' if target is not None else 'all_delivered_no_collisions') if reached else
                     'no_improvement_patience_exhausted' if patience is not None and nonimproving>=patience else 'epoch_cap'),
        no_improvement_patience=patience,consecutive_nonimproving_rounds=nonimproving,
        cost_usd=sum(costs) if all(isinstance(c,(int,float)) for c in costs) else None,
        checked_response_cost_usd=sum(costs) if all(isinstance(c,(int,float)) for c in costs) else None,
        all_http_reported_cost_usd=sum(c for c in http_costs if isinstance(c,(int,float))),
        http_requests=len(journal),unknown_charge_requests=sum(not isinstance(c,(int,float)) for c in http_costs),
        checked_parallel_response_seconds=sum(max(sum(c['seconds'] for c in calls
            if c['epoch']==h['epoch'] and c['block']==b) for b in range(len(w.groups))) for h in state['history']),
        cost_reported_calls=sum(isinstance(c,(int,float)) for c in costs),
        reasoning_tokens=sum(reasoning) if all(isinstance(c,int) for c in reasoning) else None,
        served_providers=sorted({c['served_provider'] for c in calls if c.get('served_provider')}),
        process_resumes=state['resumes'],transport_failures=sum(bool(r.get('error')) for r in rows(path/'api-requests.jsonl')))
    save(protocol/'result.json',result);state['state']='COMPLETED';save(statefile,state)
    save(path/'STATUS.json',{k:result[k] for k in ['run_status','solved','termination','F','delivered','collisions','epochs','model_calls','wall_seconds','cost_usd']})
    return result
