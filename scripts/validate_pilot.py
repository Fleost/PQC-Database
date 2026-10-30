#!/usr/bin/env python3
"""Independently reconcile completed pilot reports with their raw samples."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

def require(value, message):
    if not value:
        raise ValueError(message)

def quantile(values, p):
    ordered=sorted(values)
    position=(len(ordered)-1)*p
    low=int(position)
    return ordered[low]+(ordered[math.ceil(position)]-ordered[low])*(position-low)

def validate(root):
    status=json.loads((root/'status.json').read_text())
    require(status['status']=='passed' and status['security_selftest']=='passed', 'Pilot did not pass')
    require(len(status['reports'])==24, 'Expected 24 configurations')
    orders={}
    configurations=set()
    total=0
    for entry in status['reports']:
        directory=root/entry['path']
        report=json.loads((directory/'report.json').read_text())
        config_key=(report['quarter'],report['config']['mode'],report['config']['threshold'])
        require(config_key not in configurations, 'Duplicate configuration')
        configurations.add(config_key)
        selected={e['filename']:e for e in map(json.loads,(root/(report['quarter']+'.selected.jsonl')).read_text().splitlines())}
        require(report['coverage']['unique_documents']==len(selected), 'Wrong unique document count')
        groups=defaultdict(lambda:defaultdict(list))
        copies=Counter(); unique={}; order=[]
        with (directory/'samples.jsonl').open() as raw:
            for index,line in enumerate(raw):
                row=json.loads(line)
                require(row['measurement_index']==index and row['iteration']>=0, 'Wrong chronological index or leaked warmup')
                require(row['quarter']==report['quarter'] and row['run_id']==report['run_id'], 'Wrong provenance')
                source=selected[row['filename']]
                require(all(row[key]==source[key] for key in source), 'Raw row changed selected metadata')
                require(row['manifest_sha256']==report['manifest_sha256'], 'Wrong manifest hash')
                config=report['config']
                expected_strong=(config['mode']=='uniform_strong' or
                    (config['mode'].startswith('adaptive_') and
                     (source['payload_bytes']>config['threshold'] or
                      (config['mode']=='adaptive_size_or_importance' and source['elevation_flag']))))
                require(row['tier']==('strong' if expected_strong else 'baseline'), 'Wrong security tier')
                require(row['kem']==('ML-KEM-1024' if expected_strong else 'ML-KEM-768'), 'Wrong KEM')
                require(row['sig']==('ML-DSA-87' if expected_strong else 'ML-DSA-65'), 'Wrong signature algorithm')
                size_bin=sum(source['payload_bytes']>t*1024 for t in (10,25,50,100,250))
                labels=('overall','flagged' if source['elevation_flag'] else 'unflagged',row['tier'],'size_bin_'+str(size_bin))
                for label in labels:
                    for stage,value in row['latency_ms'].items():
                        require(math.isfinite(value) and value>=0, 'Invalid latency')
                        groups[label][stage].append(value)
                copies[row['filename']]+=1
                if row['filename'] in unique:
                    require(unique[row['filename']]['envelope']==row['envelope'], 'Inconsistent envelope sizes across repetitions')
                unique[row['filename']]=row
                order.append((row['iteration'],row['filename']))
        require(set(copies)==set(selected) and set(copies.values())=={report['iterations']}, 'Missing or repeated measured documents')
        require(len(order)==report['measured_rounds']==entry['measured_rounds'], 'Wrong measured count')
        total+=len(order)
        digest=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        require(digest==report['payload_order_sha256'], 'Wrong order checksum')
        if report['quarter'] in orders:
            require(orders[report['quarter']]==digest, 'Unpaired payload order')
        orders[report['quarter']]=digest
        for label,stages in groups.items():
            for stage,values in stages.items():
                expected=dict(count=len(values),mean=statistics.mean(values),min=min(values),max=max(values),
                              stddev=statistics.stdev(values) if len(values)>1 else None,
                              median=quantile(values,.5),p95=quantile(values,.95),p99=quantile(values,.99))
                actual=report['latency_ms'][label][stage]
                require(all((actual[k] is None if v is None else math.isclose(actual[k],v,rel_tol=1e-10,abs_tol=1e-10)) for k,v in expected.items()), 'Latency summary disagrees with raw samples')
        coverage=report['coverage']
        require(set(groups)==set(report['latency_ms']), 'Wrong latency groups')
        strong=[r for r in unique.values() if r['tier']=='strong']
        require(coverage['strong_documents']==len(strong), 'Wrong strong coverage')
        require(coverage['flagged_documents']==sum(r['elevation_flag'] for r in unique.values()), 'Wrong flagged coverage')
        require(coverage['flagged_strong_documents']==sum(r['elevation_flag'] for r in strong), 'Wrong flagged strong coverage')
        logical=report['logical_storage']
        plain=sum(r['payload_bytes'] for r in unique.values())
        stored=sum(r['envelope']['total_stored_bytes'] for r in unique.values())
        require(logical['plaintext_bytes']==plain and logical['total_stored_bytes']==stored and logical['overhead_bytes']==stored-plain, 'Wrong unique logical storage totals')
        require(coverage['strong_plaintext_bytes']==sum(r['payload_bytes'] for r in strong), 'Wrong strong byte count')
        require(math.isclose(coverage['strong_byte_fraction'],sum(r['payload_bytes'] for r in strong)/plain), 'Wrong strong byte fraction')
    expected={(q,mode,t) for q in ('2025Q4','2026Q1') for mode,t in [('uniform_baseline',None),('uniform_strong',None)]+[(mode,t*1024) for mode in ('adaptive_size','adaptive_size_or_importance') for t in (10,25,50,100,250)]}
    require(configurations==expected, 'Missing or unexpected experiment configuration')
    return dict(status='passed',configurations=24,measured_roundtrips=total,quarters=sorted(orders),independent_raw_reconciliation=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.output),indent=2))
