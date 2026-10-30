import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from app import hpc_pilot, service
from app.policy import select_policy_for_record
from scripts.prepare_faers import prepare, source_files

def source_fixture(root):
    root.mkdir()
    (root / 'DEMO25Q4.txt').write_text('primaryid$caseid$caseversion\n11$1$1\n12$1$2\n21$2$1\n')
    (root / 'OUTC25Q4.txt').write_text('primaryid$caseid$outc_cod\n12$1$DE\n12$1$LT\n')
    for table in ('DRUG','REAC','INDI','RPSR','THER'):
        (root / (table+'25Q4.txt')).write_text('primaryid$caseid$value\n12$1$example\n12$1$example\n')

def test_preparation_latest_version_source_rows_and_flags(tmp_path):
    source = tmp_path / 'input'
    source_fixture(source)
    output = tmp_path / '2025Q4'
    prepare(source, output, '2025Q4', 2)
    entries, digest = hpc_pilot.load_dataset(output)
    assert len(digest) == 64
    assert [item['primaryid'] for item in entries] == ['12','21']
    assert [item['elevation_flag'] for item in entries] == [True,False]
    doc = json.loads((output/'faers_records/12.json').read_bytes())
    assert len(doc['drugs']) == 2  # Duplicate child rows are source data.
    assert doc['demographics']['caseversion'] == '2'
    with pytest.raises(FileExistsError):
        prepare(source, output, '2025Q4', 2)
    (output/'faers_records/12.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        hpc_pilot.load_dataset(output)

def test_missing_chunk_rejected(tmp_path):
    (tmp_path/'DEMO25Q4_part002.txt').write_text('')
    with pytest.raises(ValueError,match='Missing or misnumbered'):
        source_files(tmp_path,'DEMO','2025Q4')

@pytest.mark.parametrize('size,flag,tier,reason',[(51200,False,'baseline','neither'),(51201,False,'strong','size_only'),(1,True,'strong','flag_only'),(51201,True,'strong','both')])
def test_importance_boundaries(size,flag,tier,reason):
    choice=select_policy_for_record(size,elevation_flag=flag)
    assert (choice.tier_label,choice.elevation_reason)==(tier,reason)

def test_pooled_quantiles_and_sample_stddev():
    values=[0,0,0,100]
    result=hpc_pilot.summarize(values)
    assert result['median']==0
    assert result['p95']==pytest.approx(85)
    assert result['stddev']==50
    assert hpc_pilot.summarize([])['p95'] is None

def test_diagnostic_selection_covers_strata_and_extremes():
    entries=[dict(filename=str(i),payload_bytes=size,elevation_flag=flag) for i,(size,flag) in enumerate((s,f) for s in (1,10241,25601,51201,102401,256001) for f in (False,True))]
    selected=hpc_pilot.diagnostic_sample(entries,12,42)
    assert len(selected)==12
    assert selected==hpc_pilot.diagnostic_sample(entries,12,42)
    assert len(hpc_pilot.configurations())==12

@pytest.mark.parametrize('signature,expected_alg',[(None,'ML-DSA-65'),(b'x','ML-DSA-87')])
def test_signature_requirement_before_decrypt(monkeypatch,signature,expected_alg):
    rec=SimpleNamespace(sig_alg_id='ML-DSA-65' if signature else None,signature=signature,sig_key_id='key' if signature else None)
    monkeypatch.setattr(service.db,'fetch_record',lambda *a,**k:(None,0))
    monkeypatch.setattr(service,'_row_to_record',lambda row:rec)
    monkeypatch.setattr(service.stage2,'decrypt',lambda *a,**k:pytest.fail('Must reject before decrypt'))
    with pytest.raises(ValueError):
        service.get_record(None,tenant_id='t',record_id=1,recipient_keys=None,require_signature=True,expected_sig_id=expected_alg)

def test_runner_excludes_warmups_and_counts_storage_once(tmp_path,monkeypatch):
    directory=tmp_path/'2025Q4'
    (directory/'faers_records').mkdir(parents=True)
    (directory/'faers_records/12.json').write_bytes(b'payload')
    item=dict(filename='12.json',quarter='2025Q4',caseid='1',primaryid='12',payload_bytes=7,elevation_flag=True)
    key=SimpleNamespace(pq_sk=b'key',verify_key=b'verifier')
    monkeypatch.setattr(hpc_pilot.db,'init_schema',lambda conn:None)
    monkeypatch.setattr(hpc_pilot.stage2,'generate_hybrid_recipient_keys',lambda kem:key)
    monkeypatch.setattr(hpc_pilot.stage2,'generate_sig_keys',lambda sig:key)
    calls=[]
    def put(*a,**k):
        calls.append(k)
        return len(calls),dict(encrypt=1.,sign=2.,db_insert=3.,total=6.,envelope=dict(total_stored_bytes=107))
    monkeypatch.setattr(hpc_pilot.service,'put_record',put)
    monkeypatch.setattr(hpc_pilot.service,'get_record',lambda *a,**k:(b'payload',dict(db_fetch=1.,verify=2.,decrypt=3.)))
    conn=SimpleNamespace(execute=lambda *a:SimpleNamespace(fetchone=lambda:(0,)))
    output=tmp_path/'output'; output.mkdir()
    config=dict(mode='adaptive_size_or_importance',threshold=51200)
    report=hpc_pilot.run_configuration(conn,directory,[item],config,output,'run','digest',2,1,42)
    raw=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    assert len(calls)==3 and len(raw)==2
    assert [row['iteration'] for row in raw]==[0,1]
    assert all(row['quarter']=='2025Q4' and row['tier']=='strong' for row in raw)
    assert report['logical_storage']['total_stored_bytes']==107
    assert report['coverage']['unique_documents']==1
    assert report['coverage']['flagged_strong_fraction']==1
    assert report['latency_ms']['overall']['sign']['count']==2
