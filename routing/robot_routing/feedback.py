"""Text drawings of submitted programs, with no planning or response repair.

Unknown peers never enter a drawing. Invalid programs show only a prefix that
the actual World.trace validator accepted; this prefix is never substituted
for the submitted program or used as an accepted proposal.
"""
from .world import MOVES


def _rows(world, marks):
    xmin=min(x for x,y in world.cells);xmax=max(x for x,y in world.cells)
    ymin=min(y for x,y in world.cells);ymax=max(y for x,y in world.cells)
    return dict(x_bounds=[xmin,xmax],y_bounds=[ymin,ymax],
                rows=[f'y={y:02d} '+''.join(marks.get((x,y),'.' if (x,y) in world.cells else '#')
                      for x in range(xmin,xmax+1)) for y in range(ymin,ymax+1)])


def _trace_attempt(world, robot, program):
    try:
        trace=world.trace(robot,program)
        return dict(valid=True,positions=trace.positions,command_count=len(program),
                    cost=trace.cost,arrival=trace.arrival,remaining=trace.remaining,error=None)
    except (ValueError,TypeError) as exc:
        error=str(exc)
    positions=[tuple(world.robots[robot]['start'])];count=0;failed=None
    if isinstance(program,list):
        for length in range(1,min(len(program),world.T)+1):
            try:
                prefix=world.trace(robot,program[:length])
            except (ValueError,TypeError) as exc:
                command=program[length-1];dxdy=MOVES.get(command) if isinstance(command,str) else None
                pos=positions[-1]
                target=None if dxdy is None else [pos[0]+dxdy[0],pos[1]+dxdy[1]]
                at_goal=pos==tuple(world.robots[robot]['goal'])
                legal=['wait'] if at_goal else [cmd for cmd,(dx,dy) in MOVES.items()
                                               if (pos[0]+dx,pos[1]+dy) in world.cells]
                failed=dict(tick=length,command=command,from_position=list(pos),
                            attempted_position=target,legal_commands=legal,error=str(exc))
                break
            positions=prefix.positions[:length+1];count=length
    return dict(valid=False,positions=positions,command_count=count,cost=None,
                arrival=None,remaining=None,error=error,first_invalid_command=failed)


def describe_attempt(world, group, submitted, frozen, known):
    """Render owned attempts against owned/known peers at the frozen base only."""
    attempts={i:_trace_attempt(world,i,submitted.get(world.ids[i])) for i in group}
    views={}
    for i in group:
        own=attempts[i];peers={}
        for j in range(len(world.ids)):
            if j==i:continue
            if j in group:peers[j]=attempts[j]['positions']
            elif tuple(sorted((i,j))) in known:peers[j]=world.trace(j,frozen[j]).positions
        events=[]
        for j,path in peers.items():events.extend(world.collisions(own['positions'],path,i,j))
        events.sort(key=lambda e:(e['time'],e['pair'][1]))
        program=submitted.get(world.ids[i]);positions=own['positions']
        marks={p:'*' for p in positions}
        marks[tuple(world.robots[i]['start'])]='S';marks[tuple(world.robots[i]['goal'])]='G'
        marks[positions[-1]]='P'
        bad=own.get('first_invalid_command')
        if bad and bad['attempted_position'] is not None:
            # The original wall map is also supplied; ! marks an attempted cell,
            # never a cell that the robot legally occupied.
            marks[tuple(bad['attempted_position'])]='!'
        for event in events:
            for cell in ([event['cell']] if event['kind']=='vertex' else event['edge']):
                marks[tuple(cell)]='X'
        route_map=_rows(world,marks)
        route_map['legend']='. free; # wall; * visited; S start; G goal; P last legal position; ! invalid attempted cell; X known collision (time listed separately). P/X may cover S/G.'
        steps=[dict(tick=0,command='start',position=list(positions[0]))]
        if isinstance(program,list):
            steps.extend(dict(tick=t,command=program[t-1],position=list(positions[t]))
                         for t in range(1,own['command_count']+1))
        frames=[]
        # Complete event lists and timed peer positions remain available even
        # when drawings are capped to avoid repeated co-location flooding input.
        for event in events[:3]:
            tick=event['time'];occupants={}
            occupants.setdefault(positions[tick],[]).append('B')
            for j,path in peers.items():
                if tick<len(path):occupants.setdefault(path[tick],[]).append('A' if j==event['pair'][1] else 'o')
            frame_marks={pos:(labels[0] if len(labels)==1 else 'X') for pos,labels in occupants.items()}
            frame=_rows(world,frame_marks)
            frame.update(tick=tick,event_kind=event['kind'],
                         legend={'B':f'your robot {world.ids[i]}',
                                 'A':world.ids[event['pair'][1]],
                                 'o':'other visible peers listed in peer_positions_at_tick',
                                 'X':'multiple robots at this cell'},
                         peer_positions_at_tick={world.ids[j]:list(path[tick]) for j,path in peers.items() if tick<len(path)})
            if event['kind']=='edge':
                frame['swap_edge']=event['edge']
                frame['positions_before_tick']={world.ids[i]:list(positions[tick-1]),
                                                world.ids[event['pair'][1]]:list(peers[event['pair'][1]][tick-1])}
                frame['positions_at_tick']={world.ids[i]:list(positions[tick]),
                                            world.ids[event['pair'][1]]:list(peers[event['pair'][1]][tick])}
            frames.append(frame)
        views[world.ids[i]]=dict(program_valid=own['valid'],error=own['error'],
            local_cost=own['cost'],arrival_tick=own['arrival'],remaining_distance=own['remaining'],
            last_legal_position=list(positions[-1]),legal_command_prefix_length=own['command_count'],
            first_invalid_command=bad,positions_by_tick=[list(p) for p in positions],
            trace_extent='full_horizon' if own['valid'] else 'legal_prefix_only',
            command_trace=steps,route_map=route_map,
            known_collisions=[{**event,'pair':[world.ids[j] for j in event['pair']]} for event in events],
            collision_frames=frames)
    return views
