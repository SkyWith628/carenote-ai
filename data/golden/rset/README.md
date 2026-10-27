# R-set (Sentinel 독립 평가용)

설계: `docs/sentinel-rset-design.md`. 기존 골든셋·`data/golden/FROZEN.json`과 분리되어 있다.
- `specs/probe_K.json`: 스펙 40건(dev 20 / test 20). transcript 없음, 모델에 노출된 적 없음.
- `build_specs.py.txt`: 스펙 생성표. 실행: `python3 -c "import runpy; runpy.run_path('data/golden/rset/build_specs.py.txt', run_name='__main__')"`
- test 20건 transcript는 생성 후에도 규칙 동결(`sentinel-rules-v1` 태그) 전에는 열지 않는다.
