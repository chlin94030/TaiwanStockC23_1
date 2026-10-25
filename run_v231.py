"""CLI for a local V21 pinned dataset checkout; no Streamlit UI needed."""
import argparse
from pathlib import Path
from v231_runner import run_research
from v231_engine import V23Config

def main():
    parser=argparse.ArgumentParser(description='Run research-only V23.1 on existing pinned data')
    parser.add_argument('--root',type=Path,default=Path(__file__).parent/'data'/'v21_universe_lab',help='Existing V21/V22 workspace directory; do not delete it')
    a=parser.parse_args()
    def progress(phase,message):print(phase,'|',message,flush=True)
    result=run_research(a.root,V23Config(),progress)
    print('Research completed:',result/'V23_1_VALIDATION.zip')
    print('CAUTION: EXPOSED DEVELOPMENT ONLY, NOT VERIFIED ABOVE-0050 ALPHA.')

if __name__=='__main__':main()
