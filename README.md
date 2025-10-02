# Counterfactual-Methods-for-Spectroscopy-Data
Data and code for "Counterfactual Methods for Spectroscopy Data"



## Third-party projects
This thesis work adapts code from:
- CELS - https://github.com/Luckilyeee/CELS
- Glacier (learning-time-series-counterfactuals) - https://github.com/zhendong3wang/learning-time-series-counterfactuals

The adapted code is included in this repository under the folders `CELS/` and `Glacier/`.





# To Run
## Envirionments

This project uses **two conda environments**, each provided as a YAML file.

| Methods                     | Environment file                  | Create with                                                                 |
|-----------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| CELS                        | `env-cels.yml`                    | `conda env create -f env-cels.yml`                                          |
| Glacier + RSF + Analysis    | `env-GlacierRSFAnalysis.yml`      | `conda env create -f env-GlacierRSFAnalysis.yml`                            |

Activate with:
```bash
conda activate cels
# or
conda activate GlacierRSFAnalysis
```

## Run Commands
### RSF

<table>
<tr>
<th>Mode</th>
<th>Command</th>
</tr>
<tr>
<td><strong>Global</strong></td>
<td>

```bash
python -m RSF.run_rsf_cf \
  --dataset DRS_TissueClassification \
  --repository repotwo/ucr \
  --mode global \
  --global-topk-frac 0.10 \
  --samples 250
```

</td>
</tr>
<tr>
<td><strong>Local</strong></td>
<td>

```bash
python -m RSF.run_rsf_cf \
  --dataset DRS_TissueClassification \
  --repository repotwo/ucr \
  --mode local \
  --samples 250
```

</td>
</tr>
</table>

---

### Glacier

<table>
<tr>
<th>Mode</th>
<th>Command</th>
</tr>
<tr>
<td><strong>Global</strong></td>
<td>

```bash
python -m Glacier.learning-time-series-counterfactuals.src.gc_latentcf_search \
  --dataset DRS_TissueClassification \
  --pos 1 \
  --neg 0 \
  --output DRS_TissueClassification_global.csv \
  --w-type global \
  --w-value 0.8 \
  --tau-value 0.8 \
  --lr-list 0.001 0.001 0.001 \
  --contiguity both \
  --bands-k 30 \
  --band-l 50 \
  --band-min-len 6 \
  --global-percentile 50 \
  --lam-tv 1e-3 \
  --method cnn+none
```

</td>
</tr>
<tr>
<td><strong>Local</strong></td>
<td>

```bash
python -m Glacier.learning-time-series-counterfactuals.src.gc_latentcf_search \
  --dataset DRS_TissueClassification \
  --pos 1 \
  --neg 0 \
  --output DRS_TissueClassification_local.csv \
  --w-type local \
  --w-value 0.8 \
  --tau-value 0.8 \
  --lr-list 0.001 0.001 0.001 \
  --contiguity both \
  --bands-k 30 \
  --band-l 50 \
  --band-min-len 6 \
  --global-percentile 50 \
  --lam-tv 1e-3 \
  --method cnn+none
```

</td>
</tr>
</table>

---

### CELS

<table>
<tr>
<th>Mode</th>
<th>Command</th>
</tr>
<tr>
<td><strong>Global</strong></td>
<td>

```bash
python -m CELS.CELS.main \
  --custom_npz /home/cok7/MScProject/cels_datasets/DRS_TissueClassification.npz \
  --dataset DRS_TissueClassification \
  --pname CELS_RamanCOVID19_ramanspy_preprocessed_Global \
  --mode global \
  --run_mode local \
  --dataset_type test \
  --algo cf \
  --seed_value 4 \
  --enable_seed True \
  --background_data train
```

</td>
</tr>
<tr>
<td><strong>Local</strong></td>
<td>

```bash
python -m CELS.CELS.main \
  --custom_npz /home/cok7/MScProject/cels_datasets/DRS_TissueClassification.npz \
  --dataset DRS_TissueClassification \
  --pname CELS_DRS_TissueClassification_Local \
  --mode local \
  --run_mode local \
  --dataset_type test \
  --algo cf \
  --seed_value 4 \
  --enable_seed True \
  --background_data train
```

</td>
</tr>
</table>