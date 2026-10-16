# Graph Tools Pipeline

This repository provides a Docker-based environment and utility scripts for downloading and converting graph datasets from different sources (DGL, OGB, Twitch) into multiple formats for downstream graph processing tasks.

---

## Container Setup

Start the Docker container with the required volume mounts:

```bash
docker run -it --rm \
    -v /graphs:/mnt/data \
    -v /tmp:/root/.dgl \
    graph-tools:latest bash
```

This mounts:

* `/mnt/data` → persistent dataset storage
* `/root/.dgl` → temporary DGL cache directory

---

## Dataset Download

### DGL Datasets

Available datasets:
`cora`, `reddit`, `yelp`, `flickr`, `coauthorphysics`, `amazoncomputers`

```bash
python /tools/downloaders/dgl_source_graph_downloader.py \
    -d ${DATASET} \
    -s /mnt/data/graphs/${DATASET}/dgl

mv /root/.dgl/* /mnt/data/graphs/${DATASET}/raw
```

---

### 🔹 OGB Datasets

Available datasets:
`ogbn-products`, `ogbn-proteins`, `ogbn-arxiv`, `ogbn-papers100M`, `ogbn-mag`

```bash
python /tools/downloaders/ogb_source_graph_downloader.py \
    -d ${DATASET} \
    -s /mnt/data/graphs/${DATASET}/dgl \
    -r /root/.ogb

mv /root/.ogb/* /mnt/data/graphs/${DATASET}/raw
```

---

### Twitch Dataset

```bash
python /tools/downloaders/twitch_graph_downloader.py \
    -d ${DATASET} \
    -s /mnt/data/graphs/${DATASET}/dgl

mv /root/.dgl/* /mnt/data/graphs/${DATASET}/raw
```

---

## Graph Format Conversion

After downloading, graphs can be converted into multiple formats.

---

### Edge List Format

```bash
python /tools/converters/dgl_to_edgelist.py \
    -g /mnt/data/graphs/${DATASET}/dgl/${DATASET}.dgl \
    -s /mnt/data/graphs/${DATASET}/edgelist/${DATASET}.edgelist
```

---

### METIS Format

```bash
python /tools/converters/dgl_to_metis.py \
    -g /mnt/data/graphs/${DATASET}/dgl/${DATASET}.dgl \
    -s /mnt/data/graphs/${DATASET}/metis/${DATASET}.metis
```

---

### Adjacency with Gradients (for FENNEL)

```bash
python /tools/converters/dgl_to_adj_with_grad.py \
    -g /mnt/data/graphs/${DATASET}/dgl/${DATASET}.dgl \
    -s /mnt/data/graphs/${DATASET}/adj_with_grad/${DATASET}.adj
```

---

### Cuttana Format

```bash
python /tools/converters/dgl_to_cuttana_iformat.py \
    -g /mnt/data/graphs/${DATASET}/dgl/${DATASET}.dgl \
    -s /mnt/data/graphs/${DATASET}/cuttana_iformat/${DATASET}.txt
```

---

## Output Structure

After processing, datasets are organized as follows:

```
/mnt/data/graphs/${DATASET}/
├── raw/
├── dgl/
├── edgelist/
├── metis/
├── adj_with_grad/
└── cuttana_iformat/
```

---

## Notes

* Replace `${DATASET}` with the desired dataset name.
* Ensure all output directories exist or are created before running scripts.
* The container must have access to `/mnt/data` for persistent storage.
