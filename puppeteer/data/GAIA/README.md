GAIA dataset placement

Expected default location:

`puppeteer/data/GAIA/`

Suggested download flow with ModelScope:

```bash
pip install modelscope
modelscope download --dataset gaia-benchmark/GAIA --local_dir data/GAIA
```

If the downloaded metadata is parquet only, install a parquet reader too:

```bash
pip install pandas pyarrow
```

The task loader supports common GAIA layouts such as:

- `data/GAIA/2023/validation/metadata.parquet`
- `data/GAIA/2023/test/metadata.parquet`
- `data/GAIA/validation/metadata.parquet`
- `data/GAIA/test/metadata.parquet`

It also supports:

- `metadata.level1.parquet`
- `metadata.level2.parquet`
- `metadata.level3.parquet`
- JSONL versions of the same files

If you download the dataset somewhere else, run with:

```bash
python main.py GAIA validation --gaia_data_dir /path/to/GAIA
```

Examples:

```bash
python main.py GAIA validation --level 1 --data_limit 10
python main.py GAIA validation --level 2 --gaia_data_dir data/GAIA
python main.py GAIA test --level 1 --data_limit 5
```
