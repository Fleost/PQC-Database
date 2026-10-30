import hashlib
from pathlib import Path
import shutil
import subprocess
import pytest

def package(tmp_path):
    root=tmp_path/'package'
    root.mkdir()
    script=Path(__file__).resolve().parents[1]/'cluster/install.sh'
    shutil.copy(script,root/'install.sh')
    (root/'payload').mkdir()
    chunks=[b'portable-',b'pilot-image']
    lines=[]
    for index,content in enumerate(chunks):
        name=f'payload/pilot.sif.part{index:03d}'
        (root/name).write_bytes(content)
        lines.append(hashlib.sha256(content).hexdigest()+'  '+name+'\n')
    (root/'payload/SHA256SUMS').write_text(''.join(lines))
    (root/'pilot.sif.sha256').write_text(hashlib.sha256(b''.join(chunks)).hexdigest()+'  pilot.sif\n')
    return root

def run(root):
    return subprocess.run(['bash','install.sh'],cwd=root,capture_output=True,text=True)

def test_install_reassemble_and_verify_existing(tmp_path):
    root=package(tmp_path)
    assert run(root).returncode==0
    assert (root/'pilot.sif').read_bytes()==b'portable-pilot-image'
    assert run(root).returncode==0
    (root/'pilot.sif').write_bytes(b'tampered')
    assert run(root).returncode!=0
    assert (root/'pilot.sif').read_bytes()==b'tampered'

def test_tampered_chunk_never_publishes_image(tmp_path):
    root=package(tmp_path)
    (root/'payload/pilot.sif.part001').write_bytes(b'bad')
    assert run(root).returncode!=0
    assert not (root/'pilot.sif').exists()
