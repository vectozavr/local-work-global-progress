"""Calls to the language model: the OpenRouter client, extraction of the command object from an answer, and
repetition of a request whose answer has no checkable command object (which does not count as a search step).

Standard library only. Every HTTP request and response is appended to a JSONL journal before it is used, so an
interrupted run resumes from the journal instead of paying for the same answer again.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import shlex
import threading
import time
import urllib.error
import urllib.request
import hashlib

from .world import MOVES

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_MODEL = 'openai/gpt-oss-120b'


def load_api_key(env_file):
    """Read just this key; never execute/source a dotenv file or change os.environ."""
    value = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not value and Path(env_file).is_file():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line.startswith('export '):
                line = line[7:].lstrip()
            name, separator, raw = line.partition('=')
            if separator and name.strip() == 'OPENROUTER_API_KEY':
                values = shlex.split(raw, comments=True, posix=True)
                if len(values) != 1:
                    raise ValueError('OPENROUTER_API_KEY must have one nonempty value')
                value = values[0]
    if not value or '\n' in value or '\r' in value:
        raise ValueError('A nonempty OPENROUTER_API_KEY is required for execution')
    return value


def final_answer(content):
    """Never pass unfinished or embedded reasoning to the program parser."""
    if content is None:
        return ''
    if not isinstance(content, str):
        raise ValueError('Expected a text completion')
    if '</think>' in content:
        return content.rsplit('</think>', 1)[1].strip()
    if '<think>' in content:
        return ''
    return content.strip()


class OpenRouterError(RuntimeError):
    pass


def _deadline_worker(connection, settings, prompt, seed, context):
    """The secret stays in process memory; the parent alone writes the journal."""
    model=OpenRouterModel(**settings)
    records=[]
    model._save=lambda record: records.append(model._redact(record))
    try:
        answer=model._one(prompt,seed,call_context=context)
        connection.send(dict(answer=answer,record=records[-1],error=None))
    except Exception as exc:
        connection.send(dict(answer=None,record=records[-1] if records else {},error=model._redact(str(exc))))
    finally:
        connection.close()


class OpenRouterModel:
    def __init__(self, *, api_key=None, model=DEFAULT_MODEL,
                 temperature=.6, top_p=.95, thinking=True, provider=None,
                 timeout=180., concurrency=3, send_seed=False, journal=None,
                 json_output=False, hard_deadline=None, endpoint=ENDPOINT, reasoning_effort=None):
        if concurrency < 1 or timeout <= 0 or not math.isfinite(timeout):
            raise ValueError('Concurrency and finite timeout must be positive')
        self._key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.thinking = thinking
        self.provider = provider
        self.timeout = timeout
        self.concurrency = concurrency
        self.send_seed = send_seed
        self.journal = None if journal is None else Path(journal)
        self.json_output=json_output
        self.hard_deadline=hard_deadline
        self.endpoint=endpoint
        if reasoning_effort not in (None,'low','medium','high'):
            raise ValueError('Reasoning effort must be low, medium or high')
        if reasoning_effort is not None and thinking is not True:
            raise ValueError('Reasoning effort requires thinking enabled')
        self.reasoning_effort=reasoning_effort
        if hard_deadline is not None and (hard_deadline<=0 or not math.isfinite(hard_deadline)):
            raise ValueError('Absolute deadline must be positive and finite')
        self._lock = threading.Lock()
        self.metadata = dict(backend='openrouter', model=model, endpoint=endpoint,
            temperature=temperature, top_p=top_p,
            thinking=thinking, provider_only=provider, provider_sort='throughput',
            require_parameters=True, provider_fallbacks=provider is None,
            timeout_seconds=timeout, concurrency=concurrency, streaming=False,
            retry_policy='One HTTP attempt; errors stop the arm and remain in its journal.',
            seed_policy='Sent to API; provider support required' if send_seed else
                'Coordinator seed recorded; no API determinism claimed.',
            output_accounting='Use provider usage; completion tokens include reasoning when reported.',
            execution='Independent concurrent HTTP requests, not a padded local inference batch.',
            checkpoint='OpenRouter model slug; exact HF revision/quantization not certified.',
            credential_logging=False)
        if json_output:self.metadata['response_format']='json_object'
        if hard_deadline is not None:self.metadata['absolute_client_deadline_seconds']=hard_deadline
        if reasoning_effort is not None:self.metadata['reasoning_effort']=reasoning_effort
        self.metadata['token_limit_policy']='No client token cap; token-limit parameters are unsupported'

    def payload(self, prompt, seed):
        provider = dict(sort='throughput', require_parameters=True,
                        allow_fallbacks=self.provider is None)
        if self.provider:
            provider['only'] = [self.provider]
        result = dict(model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=self.temperature, top_p=self.top_p,
            provider=provider,stream=False)
        if self.thinking is not None:
            result['reasoning']=dict(enabled=self.thinking,exclude=False)
            if self.reasoning_effort is not None:
                result['reasoning']=dict(effort=self.reasoning_effort,exclude=False)
        if self.json_output:result['response_format']=dict(type='json_object')
        if self.send_seed:
            result['seed'] = seed
        return result

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, '[REDACTED]') if self._key else value
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, dict):
            return {self._redact(k): self._redact(v) for k, v in value.items()}
        return value

    def _save(self, record):
        if self.journal is not None:
            with self._lock:
                with self.journal.open('a') as stream:
                    stream.write(json.dumps(self._redact(record)) + '\n')

    def one(self, prompt, seed, *, call_context=None):
        if self.hard_deadline is None:
            return self._one(prompt,seed,call_context=call_context)
        if not self._key:raise ValueError('No API key loaded; request preview does not execute')
        started=time.perf_counter()
        context=multiprocessing.get_context('spawn')
        parent,child=context.Pipe(duplex=False)
        settings=dict(api_key=self._key,model=self.model,
            temperature=self.temperature,top_p=self.top_p,thinking=self.thinking,provider=self.provider,
            timeout=min(self.timeout,self.hard_deadline),send_seed=self.send_seed,
            json_output=self.json_output,endpoint=self.endpoint,reasoning_effort=self.reasoning_effort)
        worker=context.Process(target=_deadline_worker,args=(child,settings,prompt,seed,call_context),daemon=True)
        record=dict(request=self.payload(prompt,seed),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        if call_context is not None:record['call_context']=call_context
        try:
            worker.start();child.close()
            remaining=max(0,self.hard_deadline-(time.perf_counter()-started))
            if not parent.poll(remaining):
                record.update(error='Absolute request deadline exceeded',client_cancelled=True,
                    usage_unknown=True,billing_unknown=True)
                raise OpenRouterError(record['error'])
            packet=parent.recv();record=packet['record'] or record
            if packet['error']:raise OpenRouterError(packet['error'])
            answer=packet['answer'];answer['worker_seconds']=answer['seconds']
            answer['seconds']=time.perf_counter()-started
            return answer
        except Exception as exc:
            record.setdefault('error',self._redact(f'{type(exc).__name__}: {exc}'))
            raise OpenRouterError(record['error']) from None
        except BaseException as exc:
            record.update(error=f'Client interrupted: {type(exc).__name__}',client_cancelled=True,
                          usage_unknown=True,billing_unknown=True)
            raise
        finally:
            if worker.pid:
                worker.join(timeout=.05)
                if worker.is_alive():worker.terminate();worker.join(timeout=1.)
            parent.close();child.close()
            record['seconds']=time.perf_counter()-started
            self._save(record)

    def _one(self, prompt, seed, *, call_context=None):
        if not self._key:
            raise ValueError('No API key loaded; request preview does not execute')
        payload = self.payload(prompt, seed)
        request = urllib.request.Request(self.endpoint,
            data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + self._key,
                     'Content-Type': 'application/json',
                     'X-OpenRouter-Title': 'Agentic robot routing'})
        started = time.perf_counter()
        record = dict(request=payload, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        if call_context is not None:
            record['call_context'] = call_context
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode('utf-8')
            record['response_body'] = self._redact(body)
            data = json.loads(body)
            record['response'] = self._redact(data)
            if not isinstance(data, dict) or data.get('error'):
                raise OpenRouterError('OpenRouter returned an API error; see the request journal')
            choices = data.get('choices') or []
            if not choices or choices[0].get('error'):
                raise OpenRouterError('OpenRouter returned no successful choice; see the request journal')
            choice = choices[0]
            message = choice.get('message') or {}
            text = final_answer(message.get('content'))
            usage = data.get('usage') or {}
            for name in ['prompt_tokens', 'completion_tokens']:
                if not isinstance(usage.get(name), int) or usage[name] < 0:
                    raise OpenRouterError('Token accounting missing or malformed; response retained in journal')
            return dict(text=self._redact(text), raw_response=self._redact(data),
                reasoning=self._redact(message.get('reasoning')),
                reasoning_details=self._redact(message.get('reasoning_details')),
                usage=usage, finish_reason=choice.get('finish_reason'),
                native_finish_reason=choice.get('native_finish_reason'),
                generation_id=data.get('id'), served_model=data.get('model'),
                served_provider=data.get('provider'), system_fingerprint=data.get('system_fingerprint'),
                cost_usd=usage.get('cost'), seconds=time.perf_counter()-started,
                max_new_tokens=None, inference_seed=seed if self.send_seed else None,
                request_payload=payload)
        except urllib.error.HTTPError as exc:
            record.update(http_status=exc.code, response_body=self._redact(exc.read().decode('utf-8', errors='replace')))
            record['error'] = f'HTTP {exc.code}'
            raise OpenRouterError(f'OpenRouter HTTP {exc.code}; see the request journal') from None
        except Exception as exc:
            record['error'] = self._redact(f'{type(exc).__name__}: {exc}')
            raise OpenRouterError(record['error']) from None
        finally:
            record['seconds'] = time.perf_counter()-started
            self._save(record)

    def batch(self, prompts, seed):
        if not prompts:
            return []
        started = time.perf_counter()
        # Preserve block ordering; each successful response is journaled immediately.
        with ThreadPoolExecutor(max_workers=min(len(prompts), self.concurrency)) as pool:
            answers = list(pool.map(lambda item: self.one(item[1], seed+item[0]), enumerate(prompts)))
        elapsed = time.perf_counter()-started
        for answer in answers:
            answer.update(batch_seconds=elapsed, batch_size=len(prompts))
        return answers

# ---------------------------------------------------------------------------------------------------------------
# Answers: the command object, answers recovered from the journal, and the format-retry wrapper

def parse_program_response(text):
    """Extract a unique complete programs object; never repair or execute content."""
    decoder=json.JSONDecoder();found=[];position=0
    while position<len(text):
        start=text.find('{',position)
        if start<0:break
        try:value,length=decoder.raw_decode(text[start:])
        except json.JSONDecodeError:position=start+1;continue
        position=start+length
        if isinstance(value,dict) and isinstance(value.get('programs'),dict):found.append(value['programs'])
    if len(found)!=1:
        raise ValueError(f'Expected one complete JSON object containing programs; found {len(found)}. Return literal command arrays, not Python code or an unfinished list.')
    return found[0]


def recovered_answer(record, seed):
    data=record['response'];choice=data['choices'][0];message=choice.get('message') or {}
    usage=data['usage']
    return dict(text=final_answer(message.get('content')),raw_response=data,
        reasoning=message.get('reasoning'),reasoning_details=message.get('reasoning_details'),usage=usage,
        finish_reason=choice.get('finish_reason'),native_finish_reason=choice.get('native_finish_reason'),
        generation_id=data.get('id'),served_model=data.get('model'),served_provider=data.get('provider'),
        system_fingerprint=data.get('system_fingerprint'),cost_usd=usage.get('cost'),seconds=record['seconds'],
        max_new_tokens=record['request'].get('max_tokens'),inference_seed=record['request'].get('seed'),request_payload=record['request'],recovered_from_journal=True)


def checkable_programs(text, owned, max_updates=None, min_updates=1):
    programs = parse_program_response(text)
    if max_updates is None:
        if set(programs) != set(owned):
            raise ValueError('The programs object must contain exactly the owned robot IDs')
    elif not (min_updates <= len(programs) <= max_updates and set(programs) <= set(owned)):
        raise ValueError(f'Return exactly {max_updates} distinct known robot IDs; omitted routes stay current' if min_updates == max_updates
                         else f'Return {min_updates} to {max_updates} distinct known robot IDs; omitted routes stay current')
    for rid, commands in programs.items():
        if not isinstance(commands, list):
            raise ValueError(f'{rid}: expected a literal command array')
        if any(not isinstance(c, str) or c not in MOVES for c in commands):
            raise ValueError(f'{rid}: commands must be up/down/left/right/wait strings')
    return programs


def retry_seed(seed, index, policy):
    """Reproducible resampling avoids endlessly repeating one bad seeded JSON."""
    if policy == 'identical' or index == 0:
        return seed
    return int.from_bytes(hashlib.sha256(f'{seed}:retry:{index}'.encode()).digest()[:4], 'big') % (2**31)


class CheckableModel:
    def __init__(self, backend, format_attempts=8, transport_attempts=3, retry_seed_policy='identical'):
        if retry_seed_policy not in ('identical', 'resample'):
            raise ValueError('Unknown retry seed policy')
        self.backend = backend
        self.format_attempts = format_attempts
        self.transport_attempts = transport_attempts
        self.retry_seed_policy = retry_seed_policy
        self.journal = backend.journal
        self.metadata = {**backend.metadata,
            'schema_retry_policy': 'Identical prompt and settings; failed schema is not a search step',
            'retry_seed_policy': retry_seed_policy,
            'schema_attempt_cap': format_attempts, 'transport_attempt_cap': transport_attempts,
            'timing': 'Accepted-schema response latency excludes preceding failed requests; total wall/cost retained'}

    def payload(self, prompt, seed):
        return self.backend.payload(prompt, seed)

    def one(self, prompt, seed, *, call_context=None):
        task = json.JSONDecoder().raw_decode(prompt.split('\nTASK_JSON\n', 1)[1])[0]
        owned = task['owned']
        max_updates = task.get('max_updated_robots')
        min_updates = task.get('min_updated_robots', 1)
        context = call_context or {}
        existing = [r for r in rows(self.journal)
                    if r.get('call_context', {}).get('search_context') == context]
        excluded_seconds = 0.
        schema_failures = 0
        transport_failures = 0
        transport_failures_total = 0
        reset_through = -1
        truncated_retry_index = None
        # A diagnosed provider truncation may receive ONE explicit physical
        # retry. This does not change the request, token policy or search budget.
        # A repeated truncation still stops before another paid request.
        truncfile = Path(self.journal).parent/'truncation-resume.json' if self.journal else None
        if truncfile and truncfile.exists():
            grants = [g for g in json.loads(truncfile.read_text())['grants']
                      if g['search_context'] == context]
            if len(grants)>1:
                raise ValueError('Only one explicit truncation renewal per logical step is allowed')
            if grants:
                g=grants[0];i=g['request_index']
                if i<0 or i>=len(existing):
                    raise ValueError('Truncation renewal needs an archived request')
                r=existing[i]
                request_sha=hashlib.sha256(json.dumps(r['request'],sort_keys=True,
                    separators=(',',':')).encode()).hexdigest()
                choice=(r.get('response',{}).get('choices') or [{}])[0]
                if (r.get('error') or choice.get('finish_reason')!='length'
                        or request_sha!=g['request_sha256']):
                    raise ValueError('Truncation renewal does not match its diagnosed request')
                truncated_retry_index=i
        # An operator can renew ONE exhausted 429 transport budget after a
        # cooldown. Raw errors and schema counters remain unchanged, and the
        # request index (hence retry seed) continues rather than starting over.
        grantfile = Path(self.journal).parent/'transport-resume.json' if self.journal else None
        if grantfile and grantfile.exists():
            grants = [g for g in json.loads(grantfile.read_text())['grants']
                      if g['search_context'] == context]
            if len(grants)>1:
                raise ValueError('Only one explicit transport renewal per logical step is allowed')
            if grants:
                reset_through = grants[0]['reset_after_request_index']
                before = existing[:reset_through+1]
                failures = [r for r in before if r.get('error')]
                if len(before)!=reset_through+1 or len(failures)<self.transport_attempts:
                    raise ValueError('Transport renewal requires an archived exhausted transport budget')
        # Reuse responses already written before a crash, including failed
        # schema attempts. Never pay for the same completed request again.
        for index, r in enumerate(existing):
            request_seed = retry_seed(seed, index, self.retry_seed_policy)
            if r['request'] != self.payload(prompt, request_seed):
                raise ValueError('Resume prompt differs from its recorded request')
            if r.get('error'):
                transport_failures_total += 1
                transport_failures += int(index>reset_through)
                excluded_seconds += r['seconds']
                continue
            answer = recovered_answer(r, request_seed)
            try:
                checkable_programs(answer['text'], owned, max_updates, min_updates)
            except ValueError:
                if answer.get('finish_reason') == 'length' and index != truncated_retry_index:
                    raise RuntimeError('Provider truncated the reply. Inspect its token policy; do not repeat the same truncated request.')
                schema_failures += 1
                excluded_seconds += r['seconds']
                continue
            answer.update(schema_valid=True, schema_retries=schema_failures,
                          transport_retries=transport_failures_total, excluded_request_seconds=excluded_seconds)
            return answer
        request_index = len(existing)
        while schema_failures < self.format_attempts and transport_failures < self.transport_attempts:
            if truncated_retry_index is not None and request_index>truncated_retry_index+1:
                raise RuntimeError('The single diagnosed-truncation retry is exhausted; inspect before continuing')
            request_seed = retry_seed(seed, request_index, self.retry_seed_policy)
            request_context = dict(search_context=context, request_index=request_index,
                                   logical_seed=seed, retry_seed_policy=self.retry_seed_policy)
            request_index += 1
            tic = time.perf_counter()
            try:
                answer = self.backend.one(prompt, request_seed, call_context=request_context)
            except OpenRouterError as exc:
                transport_failures += 1
                transport_failures_total += 1
                excluded_seconds += time.perf_counter() - tic
                print(json.dumps(dict(event='transport_retry', search_context=context,
                                      transport_failures=transport_failures)), flush=True)
                if transport_failures<self.transport_attempts:
                    time.sleep(min(60.,30.*2**(transport_failures-1)))
                continue
            try:
                checkable_programs(answer['text'], owned, max_updates, min_updates)
            except ValueError as exc:
                if answer.get('finish_reason') == 'length':
                    raise RuntimeError('Provider truncated the reply. Inspect its token policy; do not repeat the same truncated request.')
                schema_failures += 1
                excluded_seconds += answer['seconds']
                print(json.dumps(dict(event='format_retry', search_context=context,
                                      schema_failures=schema_failures, error=str(exc))), flush=True)
                continue
            answer.update(schema_valid=True, schema_retries=schema_failures,
                          transport_retries=transport_failures_total, excluded_request_seconds=excluded_seconds)
            return answer
        raise RuntimeError('Identical-request retry cap reached; raw responses are retained. Inspect before continuing.')


def openrouter_model(settings, api_key, journal):
    """Client with the sampling settings of a campaign, wrapped so that malformed answers are re-asked."""
    backend = OpenRouterModel(api_key=api_key, model=settings['model'], provider=settings['provider'], thinking=True,
                              reasoning_effort=settings.get('reasoning_effort'), temperature=settings['temperature'],
                              top_p=settings['top_p'], timeout=settings['timeout'], concurrency=settings.get('width', 3),
                              send_seed=True, journal=journal)
    return CheckableModel(backend, settings['format_attempts'], settings['transport_attempts'], settings['retry_seed_policy'])
