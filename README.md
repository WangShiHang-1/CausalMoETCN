# CausalMoETCN

**Multivariate KPI response prediction under operational interventions in core networks**

CausalMoETCN predicts how network key performance indicators (KPIs) evolve during a known operation plan and the subsequent recovery period. It combines a causal temporal convolutional network (TCN), difference features, feature-wise linear modulation (FiLM), and a mixture of causal convolutional experts.

This document supplements the manuscript **“CausalMoETCN: Multivariate KPI Response Prediction under Operational Interventions in Core Networks.”** It explains the input interface, preprocessing, temporal dependence, training objectives, dataset-specific protocols, and baseline adaptations that are abbreviated in the paper.

The architecture and core-network data interface below have been checked against the implementation. Numerical results are quoted from the manuscript; creating this document did not rerun the reported experiments. Dataset-specific run records take precedence over generic or historical runner defaults when reproducing a result.

## Contents

- [Task and interpretation](#task-and-interpretation)
- [Core-network data](#core-network-data)
- [Preprocessing and missing observations](#preprocessing-and-missing-observations)
- [Architecture and tensor shapes](#architecture-and-tensor-shapes)
- [Operation-prefix constraint](#operation-prefix-constraint)
- [Training and evaluation](#training-and-evaluation)
- [Public datasets and their protocols](#public-datasets-and-their-protocols)
- [Baseline adaptations](#baseline-adaptations)
- [Reported results and ablations](#reported-results-and-ablations)
- [Model interface example](#model-interface-example)
- [Implementation references and reproduction status](#implementation-references-and-reproduction-status)

## Task and interpretation

At a forecast origin, the model receives historical KPIs, historical operational covariates, historical operations, and a specified future operation plan. It predicts a multivariate KPI trajectory over the forecasting horizon.

A plan records operation types, scheduled execution times, and target service groups. The requested output is a **response trajectory**, including persistent effects and recovery, rather than an operation label or a fault diagnosis.

### What is known at prediction time?

| Information | Available to the model? | Role |
|---|---|---|
| Historical KPIs | Yes | Previous network response |
| Historical operational covariates | Yes | Previous load, session, and system state |
| Historical observation indicators and observation ages | Yes | Availability information embedded in historical features |
| Historical operations | Yes | Previous operation conditions |
| Future operation plan | Yes, by task assumption | Conditioning information |
| Future KPI values | No | Training targets and evaluation only |
| Future observed covariates | No | Not inputs to this model |
| Future target observation masks | No | Loss and metric calculation only |
| Sample identifiers, pair identifiers, and matched control trajectories | No | Dataset bookkeeping or analysis |

A scheduled plan and the realized network response are different objects. Delayed execution, recovery, and persistent effects can appear in the observed KPIs even after the scheduled operation interval ends.

### Causal interpretation

Let $`\mathcal H`$ denote historical observations and let $`Y^{\mathrm{fut}}[a]`$ be the potential KPI trajectory under plan $`a`$ and the specified execution mechanism. The paper defines the target conditional expected response as

```math
\mu(\mathcal H,a)
=
\mathbb E[Y^{\mathrm{fut}}[a]\mid\mathcal H].
```

Identification from observational data requires:

1. **Consistency:** the observed trajectory equals the potential trajectory under the adopted plan.
2. **Conditional exchangeability at the plan level:**

   ```math
   Y^{\mathrm{fut}}[a]\ \perp\!\!\!\perp\ A^{\mathrm{fut}}\mid\mathcal H.
   ```

3. **Positivity:** the target plan lies within the conditional support for the observed historical state.

Under these conditions,

```math
\mu(\mathcal H,a)
=
\mathbb E[Y^{\mathrm{fut}}\mid\mathcal H,A^{\mathrm{fut}}=a].
```

The temporal restriction additionally uses **non-anticipation**: for the same history, plans that agree through step $`j`$ have the same potential responses through that step.

The network's causal convolutions enforce a direction of temporal dependence. They do not establish exchangeability or positivity. Expert routing weights are also not estimates of causal effects. Matched no-operation windows are observed comparison samples, not observed counterfactual outcomes for the same operation window.

The loss used to train a response predictor need not have the original-scale conditional mean as its minimizer. In particular, the transformed-scale SmoothL1 objective described below does not generally recover that mean exactly.

## Core-network data

The testbed combines a Kubernetes-based core network and a radio access network (RAN), with four base stations and 40 user equipment (UE) devices. Data are sampled on a one-second grid.

The targets, in channel order, are:

1. N3 uplink bit rate;
2. N3 downlink bit rate;
3. N6 aggregate bit rate.

### Operation encoding

The operation input is an 11-channel multi-hot vector. The first nine channels encode operation types; the last two identify service groups.

| Index | Stored field | Meaning |
|---:|---|---|
| 0 | `core_n3_block` | N3 link blocking |
| 1 | `core_n3_delay` | N3 latency injection |
| 2 | `core_n3_loss` | N3 packet-loss injection |
| 3 | `core_n4_block` | N4 link blocking |
| 4 | `core_n6_block` | N6 link blocking |
| 5 | `core_upf_stop` | User plane function shutdown |
| 6 | `ran_gnb_restart` | Base-station restart |
| 7 | `ran_ue_recreate` | UE recreation |
| 8 | `ran_ue_scale_down` | UE scale-down |
| 9 | `tier_gold` | Gold service-group indicator |
| 10 | `tier_silver` | Silver service-group indicator |

Operation types and service-group indicators have different roles. When identifying an active operation interval for a plot, use operation-type channels rather than treating a service-group indicator as an operation.

The manuscript describes 12 joint operation templates and a no-operation template, with scheduled durations of 120 or 180 seconds. The complete template-to-channel mapping should be obtained from the dataset manifest; the channel list alone does not specify every joint template.

### Historical feature layout

The implementation packs the complete historical input into `history[B, L, 68]`. The value 68 is the width of the **packed history tensor**, not the number of additional covariates in the paper's separate $`X`$, $`A`$, and $`Y`$ notation.

Python slices below use an inclusive start and exclusive end.

| Slice | Channels | Contents |
|---|---:|---|
| `0:11` | 11 | Historical operation encoding |
| `11:29` | 18 | Operational covariate values |
| `29:47` | 18 | Covariate observation indicators |
| `47:65` | 18 | Log-transformed observation ages |
| `65:68` | 3 | Historical relative-log KPI values |

The 18 covariates are session establishment success rate, session failure count, round-trip time, jitter, packet-loss rate, business completion rate, active session count, pending session count, session request count, registration success rate, registration request count, active RAN PDU count, ready base-station count, observed UE count, N4 bit rate, host CPU utilization, host memory utilization, and one-minute host load average.

Historical KPI validity masks are stored separately for the difference loss. They are distinct from the 18 covariate observation-indicator channels in the model input.

### Windows and data split

The archived core-network dataset contains 1,006 windows. Each stored window has 300 historical steps and 300 future steps. The manuscript evaluates forecasting horizons of 100, 200, and 300 steps.

| Split | Collection batches | Windows | Operation windows | No-operation controls |
|---|---|---:|---:|---:|
| Training | D01–D04 | 576 | 288 | 288 |
| Validation | D05 | 144 | 72 | 72 |
| Test | D06–D07 | 286 | 143 | 143 |

The split is determined by collection batch, rather than by randomly redistributing windows. Windows do not cross collection-file boundaries. The original event-anchored archive has zero raw historical operation channels; the interface can represent historical operations, but this archive alone does not evaluate all possible historical-operation patterns.

For the inspected 100- and 200-step runners, the future plan, target, and future validity mask are truncated to the requested horizon before training. The history remains 300 steps, and a horizon-specific model is instantiated and trained. These runners therefore do more than crop an already produced 300-step prediction.

## Preprocessing and missing observations

### Relative-log KPI transform

For each window and target channel, let $`b_c`$ be the median of the genuinely observed throughput values in the 60 seconds preceding the forecast origin. The archived transformation is

```math
u_{i,c}=\log(1+y_{i,c})-\log(1+b_c),
```

where throughput is represented in bits per second. The transformed values are then standardized using the saved training statistics:

```math
z_{i,c}=\frac{u_{i,c}-m_c}{s_c}.
```

Historical KPI channels and future targets use the same target-channel statistics. Historical and future operation channels also use the same operation-channel statistics. The loader reads the saved scaler and does not refit it on validation or test data.

To convert a standardized prediction back to Mbps:

```python
relative_log = prediction * target_scale + target_mean
prediction_mbps = (
    np.expm1(relative_log + np.log1p(baseline_bps)[:, None, :]) / 1e6
)
```

The inspected inverse transformation does not clip negative predictions. Any alternative clipping policy would need to be documented because it changes errors and plotted trajectories.

### Missing historical inputs

Historical covariates and KPIs are forward-filled using only earlier observations. Before the first available observation, the construction uses a training-data median fallback. Future observations are not used to fill historical inputs.

Covariate availability is represented by observation indicators and

```math
\log(1+\min(\mathrm{age},86400)),
```

where age is measured in seconds. These channels distinguish an available observation from an old or imputed value.

### Missing future targets

Unobserved future targets remain `NaN` in the archive, with a separate Boolean validity mask. Finite placeholders may be used internally for arithmetic, but losses and metrics must include only valid targets.

For a lagged difference, **both endpoints must be valid**. A finite imputed value is not sufficient to make a historical endpoint valid for supervision.

### Raw zero and standardized zero are different

A raw no-operation plan maps to

```math
a_{\mathrm{no\ operation}}^{\mathrm{std}}=-m_a/s_a.
```

It is generally not an all-zero vector after standardization. The paper's **w/o Future Plan** ablation explicitly uses an all-zero input in the standardized space. That ablation should not be described as inserting the transformed raw no-operation plan.

## Architecture and tensor shapes

The following flow describes the full model:

```text
Historical observations --> history projection + segment embedding --+
                                                                    +--> time concatenation
Known future plan --------> plan projection + segment embedding -----+
                                                                           |
Historical KPI differences + operation differences --> gated event features |
                                                                           v
                                                                    causal TCN
                                                                           |
Operation values + signed/absolute differences --> FiLM parameters ---------+
                                                                           v
                                                                  modulated states
                                                                     /          \
                                                full timeline --> experts       future segment --> router
                                                                     \          /
                                                               weighted expert features
                                                                           |
                                                   residual addition + normalization
                                                                           |
                                                                     output head
                                                                           |
                                                                  future KPI responses
```

### 1. Projection and concatenation

Historical observations and future plans use separate $`1\times1`$ projections and learnable segment embeddings. Their projected representations are concatenated **along time**, producing a sequence of length $`L+P`$.

This operation is different from the relative-index channel concatenation used by some adapted forecasting baselines.

### 2. Difference features and gated event encoding

The backward difference is the current value minus the previous value, with the first position set to zero. Operation differences are computed on the concatenated historical/future timeline:

```math
\Delta \bar A_L=\bar A^{\mathrm{fut}}_0-\bar A^{\mathrm{hist}}_{L-1}.
```

The boundary difference is retained; it is not reset to zero at the forecast origin. KPI differences are computed from the preprocessed historical KPI values. Their future segment is filled with zeros because future KPI changes are unknown.

The event input contains signed and absolute KPI and operation differences:

```math
D_i=[\Delta\bar Y_i,\ |\Delta\bar Y_i|,\ \Delta\bar A_i,\ |\Delta\bar A_i|].
```

For three KPI channels and 11 operation channels, it has 28 channels. Two bias-free $`1\times1`$ convolutions with an intervening GELU activation map it to the hidden width. This construction satisfies $`\phi_e(0)=0`$.

A sigmoid gate combines event features with the projected timeline. Historical availability indicators enter through the historical projection, rather than being treated as KPI differences.

### 3. Temporal backbone

The inspected core-network configuration uses:

| Setting | Value |
|---|---|
| Hidden channels | 256 |
| Residual blocks | 12 |
| Convolutions per block | 2 |
| Convolution kernel size | 3 |
| Dilations | `1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8` |
| Dropout | 0.05 |
| Normalization | Layer normalization over channels independently at each time step |
| Backbone receptive field | 1,081 steps |

The receptive field above is that of the TCN backbone, computed as $`1+2(k-1)\sum_\ell d_\ell`$. Difference features and the expert convolution add dependencies beyond the backbone itself; 1,081 should not be interpreted as the full model's exact receptive field.

### 4. Operation-conditioned FiLM

The FiLM condition vector contains operation values, signed operation differences, and absolute operation differences: 33 channels for the core-network configuration.

The modulation is

```math
\gamma_i=1+\tanh(r_{\gamma,i}),\qquad
H'_i=\gamma_i\odot H_i+\beta_i.
```

The scale lies in $`(0,2)`$ and can vary around one. FiLM acts on the full historical/future timeline after the TCN and before the expert module.

### 5. Expert fusion

Four experts have identical architectures and independent parameters. Each uses:

```text
1x1 channel reduction -> GELU -> left-padded causal convolution
-> channel normalization -> GELU -> dropout -> 1x1 channel expansion
```

The bottleneck width is 64 and the expert convolution kernel size is three.

The experts process the full modulated timeline before their future outputs are selected. The router reads the future segment of the modulated states and produces a softmax distribution over experts for every sample and future time step.

- All experts contribute: this is dense soft routing, not top-k expert selection.
- One expert weight is shared across the hidden channels at a given sample/time position.
- Weights are not separately predicted for each KPI.
- Expert identities are not supervised as “fault,” “normal,” or “recovery” categories.

The weighted expert features are added to the modulated state, divided by $`\sqrt{2}`$, normalized, and passed to a shared output head. All future KPI values are produced in one forward pass, without autoregressive rollout.

### Shape reference

Here $`B`$ is batch size, $`C=256`$, and $`K=4`$ for the inspected core-network configuration.

| Tensor | Shape |
|---|---|
| Packed historical input | `[B, L, 68]` |
| Future plan | `[B, P, 11]` |
| Projected timeline | `[B, L+P, C]` |
| Difference features | `[B, L+P, 28]` |
| FiLM condition vector | `[B, L+P, 33]` |
| Modulated states | `[B, L+P, C]` |
| Routing weights | `[B, P, K]` |
| Predicted KPIs | `[B, P, 3]` |

The implementation transposes hidden tensors to `[B, C, T]` for convolution. The public forward interface uses `[B, T, C]`.

## Operation-prefix constraint

With fixed history, parameters, and deterministic inference, predictions through future step $`j`$ depend only on operation tokens through $`j`$:

```math
\widehat Y^{\mathrm{fut}}_{0:j}
=
F_{\theta,0:j}(\mathcal H,A^{\mathrm{fut}}_{0:j}).
```

Editing only the plan after $`j`$ must leave earlier predictions unchanged, up to numerical tolerance.

This property depends on the complete computation:

- backward differences must not inspect later time steps;
- convolutions must use left padding;
- normalization must operate independently at each time position;
- projections, FiLM, routing, and output mapping must be local in time;
- preprocessing must not derive earlier features from the later plan suffix.

The implementation exposes a historical `groupnorm` option. That option normalizes over time and does **not** satisfy the same prefix restriction. The described full model uses `channel_layernorm`.

A prefix check is performed in evaluation mode with dropout disabled. It verifies an architectural dependence property, not the causal identification assumptions stated above.

## Training and evaluation

### Level and difference supervision

For the core-network objective, the level loss is masked SmoothL1 on standardized targets. A second masked SmoothL1 term supervises lag-$`d`$ increments, with $`d=5`$ steps at the one-second sampling interval.

For either a predicted or observed future sequence $`z`$, the initial increments use a historical anchor:

```math
\Delta_d z_{j,c}=z_{j,c}-\bar Y^{\mathrm{hist}}_{L-d+j,c},\qquad 0\le j<d.
```

Once a lagged future value is available, the increments use that value:

```math
\Delta_d z_{j,c}=z_{j,c}-z_{j-d,c},\qquad d\le j<P.
```

Predictions and targets share the same historical anchor. For the first $`d`$ positions, subtracting the two increments cancels that anchor, leaving the corresponding level error. Later increment errors compare prediction errors separated by $`d`$ steps.

The core-network objective uses SmoothL1 threshold $`\tau=0.5`$ and

```math
J=J_{\mathrm{level}}+\lambda J_\Delta,\qquad \lambda=1.
```

Each term averages over its own valid entries; a term is zero when its valid set is empty.

### Training loss versus checkpoint selection

The core-network validation selection score is

```math
S_{\mathrm{val}}
=
\mathrm{MAE}_{\mathrm{val}}
+0.25\,\mathrm{MAE}_{\Delta_5,\mathrm{val}}.
```

The coefficient 0.25 belongs to **checkpoint selection**, not to the training loss. It must not replace the training difference-loss weight of one.

### Core-network settings stated in the manuscript

| Setting | Manuscript value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 10⁻⁴ |
| Batch size | 144 |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Number of random seeds | 5 |
| Historical length | 300 |
| Forecasting lengths | 100, 200, 300 |

**Protocol provenance:** the inspected 100/200-step coordinator uses seeds 2026–2030 and the 100-epoch/patience-10 settings, with a separate 10-epoch budget for TimeMixer++. Older 300-step comparison runners use three seeds, a 120-epoch cap, and patience 20. These historical settings are not interchangeable with the manuscript table. A complete reproduction must identify the exact run bundle behind each reported row.

### Metric scale and aggregation

Core-network MAE and MSE are computed over valid target entries in the **standardized relative-log space**. They are not errors in Mbps.

For each public dataset, targets are evaluated on its own training-standardized scale, with equal-weight averaging across the three targets. These overall values do not permit direct comparison of physical prediction errors across datasets with different units or transformations.

Reported standard deviations describe variation across the random seeds of the corresponding experiment. They are not confidence intervals and do not measure variation across independent network deployments.

## Public datasets and their protocols

Models are trained and evaluated independently on each dataset. These experiments are not zero-shot transfer from the core-network model.

| Dataset | Sampling | History / prediction | Targets | Future conditioning |
|---|---|---|---|---|
| TelecomTS | 100 ms | 64 / 64 | Uplink BLER, uplink SNR, uplink PRB utilization | Jamming-state annotations |
| AIOps2025 HipsterShop | 1 minute | 30 / 30 | Request volume, response time, error rate | Fault type, injection level, start/end annotations |

The future annotations are treated as known at prediction time. For these public-data experiments, that is a retrospective annotation-availability assumption; it does not establish that the same information would be available before a real unexpected fault.

### TelecomTS

The inspected task uses a single Jamming/YouTube source sequence, with 180 training, 51 validation, and 55 test windows. The split is temporal, with overlapping raw support removed at split boundaries. This is a defined subset/task, not a claim to cover the entire TelecomTS benchmark.

The manuscript's CausalMoETCN result, MAE **0.2376 ± 0.0070** and MSE **0.4127 ± 0.0051**, matches the 20-epoch row of the mixed-loss budget report:

| Setting | Matching run |
|---|---|
| Objective | 0.25 × MSE + 0.75 × SmoothL1 (τ = 1) |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 5 × 10⁻⁴ |
| Dropout | 0.05 |
| Seeds | 2026–2030 |
| Budget | 20 epochs |
| Checkpoint selection | Minimum validation MAE within that budget |

This result therefore does not use the core-network level-plus-five-step-difference objective. The archived study evaluated multiple budgets; the table above identifies the particular row used by the manuscript, without asserting that it was the only budget examined.

Source record: `public_datasets_workspace/experiments/telecom_mse_tuning/mixed_epoch_budgets/REPORT.md`.

### AIOps2025

The inspected HipsterShop experiment uses June 6–7, 2025 for training, June 9 for validation, and June 17 for testing, giving 388/207/207 windows. Targets use a log1p transform followed by training-statistic standardization.

The manuscript's CausalMoETCN result, MAE **0.2776 ± 0.0018** and MSE **0.4929 ± 0.0044**, matches the selected E4 configuration:

| Setting | Matching run |
|---|---|
| Hidden channels / experts | 64 / 2 |
| Objective | Level SmoothL1 + 0.05 × one-step difference SmoothL1 |
| SmoothL1 threshold | 0.1 |
| Learning rate | 2 × 10⁻⁴ |
| Weight decay | 10⁻⁴ |
| Dropout | 0.15 |
| Seeds | 2026, 2027, 2028 |
| Budget | 20 epochs |

The selection study evaluated six candidates on training/validation data, screened them with two seeds, and added a third seed for the two finalists. It used a normalized validation criterion

```math
S=\max\left(
\frac{\mathrm{MAE}_{\mathrm{val}}}{0.09190142362154563},
\frac{\mathrm{MSE}_{\mathrm{val}}}{0.03600562058092182}
\right),
```

with lower values preferred. Checkpoint selection used the saved epoch-10/epoch-20 validation endpoints. The final configuration was compared against the existing reference before the new winner was evaluated on the test date.

The matching record contains **three seeds**, rather than five. The test date had been used in earlier project experiments, so it should not be described as a previously unseen test date for the project as a whole.

Source record: `public_datasets_workspace/experiments/aiops2025_dynamics20_search_v1/REPORT.md`.

## Baseline adaptations

The baselines listed in the manuscript are task-adapted implementations. Their interfaces should not be described collectively as “the same future-plan channels added to every model.”

The following details are verified in the inspected **core-network comparison code**:

| Model | Verified task adaptation |
|---|---|
| iTransformer | Learnable projection of the 11 future-plan channels, followed by channel concatenation with the packed historical sequence |
| TimeMixer++ | PyPOTS TimeMixer++ backbone, projected future-plan concatenation, and a three-target readout |
| TimeXer | Projected future-plan concatenation; the documented adapter uses a separate MS backbone for each KPI |
| TiDE | Historical KPI channels plus historical/future operation channels through the known-covariate interface |
| Wave-CMNet | Local adapter receives packed historical information and the future operation plan; the experiment notes identify a J=1 Haar implementation |
| Causal Transformer | Local continuous-treatment task adaptation, not an unchanged reproduction of the original paper |
| CausalCPC | Local architecture/task adaptation with CPC/InfoMax components, not an unchanged reproduction of the original paper |

For the projection-and-concatenation forecasters, concatenation pairs relative positions from historical and future intervals. It does not mean that the paired entries occurred at the same physical time. For 100/200-step runs, the projected plan is zero-padded to 300 positions; no operation tokens beyond the requested horizon are supplied.

**TiDE receives a different historical information set.** In the inspected core-network adapter, it uses three historical KPIs and 11 historical/future operation channels. The other 54 historical covariate/availability channels are omitted because its selected interface expects covariates for both past and future. This difference must accompany interpretation of the comparison.

Training budgets and auxiliary losses also vary between retained experiments. In particular, the inspected runners give TimeMixer++ a separate epoch budget, and some local causal-model adaptations retain auxiliary objectives. Shared MAE/MSE definitions do not imply identical optimization budgets or identical input information.

## Reported results and ablations

These are the CausalMoETCN entries reported in the current manuscript, rather than newly generated measurements.

| Dataset | Horizon | MAE ↓ | MSE ↓ |
|---|---:|---:|---:|
| Core network | 100 | 0.4822 ± 0.0054 | 0.4815 ± 0.0121 |
| Core network | 200 | 0.4909 ± 0.0046 | 0.4773 ± 0.0065 |
| Core network | 300 | 0.4738 ± 0.0011 | 0.4490 ± 0.0016 |
| TelecomTS | 64 | 0.2376 ± 0.0070 | 0.4127 ± 0.0051 |
| AIOps2025 | 30 | 0.2776 ± 0.0018 | 0.4929 ± 0.0044 |

Values are on the dataset-specific evaluation scales described above. Different forecast horizons and run budgets should not be interpreted as a controlled, per-time-step error-growth experiment.

### Core-network ablations at 300 steps

| Variant | MAE ↓ | MSE ↓ |
|---|---:|---:|
| Full | 0.4738 ± 0.0011 | 0.4490 ± 0.0016 |
| w/o Future Plan | 0.5342 ± 0.0074 | 0.5580 ± 0.0182 |
| w/o Event | 0.4810 ± 0.0074 | 0.4626 ± 0.0114 |
| w/o FiLM | 0.4961 ± 0.0050 | 0.4789 ± 0.0106 |
| w/o MoE | 0.5012 ± 0.0046 | 0.4806 ± 0.0034 |
| Uniform Router | 0.4772 ± 0.0062 | 0.4557 ± 0.0105 |
| w/o J<sub>Δ</sub> | 0.4928 ± 0.0058 | 0.4830 ± 0.0100 |

Variant definitions:

- **Full:** Complete model.
- **w/o Future Plan:** Replace future plan with standardized zeros consistently across plan-dependent branches.
- **w/o Event:** Remove event-feature injection.
- **w/o FiLM:** Use TCN states as expert and router inputs.
- **w/o MoE:** Use modulated states as output-head inputs.
- **Uniform Router:** Retain all experts with weights fixed at 1/K.
- **w/o J<sub>Δ</sub>:** Train with the level loss only.

Dynamic routing has slightly lower mean errors than uniform weighting in this table. The table alone does not establish statistical significance or an interpretable assignment of network mechanisms to experts.

### Qualitative response figure

The response plot is a selected low-error example, not a randomly selected representative of the full test set. Its recorded provenance identifies sample 864, collection batch D07, and seed 2026. The candidate selection favored fully observed windows with lower error than CausalCPC on each KPI. The full 300-step horizon and all three KPIs were retained.

The yellow region marks the scheduled operation interval, and time zero is the forecast origin. The plot illustrates local trajectories; the aggregate tables provide the overall evaluation.

## Model interface example

The core model is implemented in `models/causal_moe_tcn.py`. Its interface is:

```python
prediction = model(history, future_treatment)
prediction, routing = model.forward_with_routing(history, future_treatment)
```

The following CPU example uses synthetic standardized tensors. It checks shapes and the operation-prefix property; it is not an accuracy evaluation or a paper-reproduction command.

```python
import torch
from models.causal_moe_tcn import CausalMoETCN

torch.set_num_threads(2)
torch.manual_seed(2026)

model = CausalMoETCN(pred_len=8).eval()
history = torch.randn(1, 32, 68)
plan = torch.randn(1, 8, 11)

with torch.no_grad():
    prediction, routing = model.forward_with_routing(history, plan)

    changed_plan = plan.clone()
    changed_plan[:, 4:, :] += 0.5
    changed_prediction = model(history, changed_plan)

assert prediction.shape == (1, 8, 3)
assert routing.shape == (1, 8, 4)
torch.testing.assert_close(
    routing.sum(dim=-1), torch.ones(1, 8), atol=1e-6, rtol=1e-6
)
torch.testing.assert_close(
    prediction[:, :4], changed_prediction[:, :4], atol=1e-6, rtol=1e-6
)
print("Shape and operation-prefix checks passed.")
```

For the archived core-network dataset, `experiments/dataset.py` loads the 300/300 data and applies the saved scaler. The loader's `h` and `p` arrays are model inputs; `y`, `m`, and `hm` are targets and validity masks. Avoid fitting another scaler or substituting archived future covariates for model inputs.

## Implementation references and reproduction status

### Inspected code layout

| File | Purpose |
|---|---|
| `models/causal_moe_tcn.py` | Full model, difference features, FiLM, experts, routing |
| `baselines_adapted/conditional_tcn.py` | Causal convolution and residual TCN components |
| `configs/causal_moe_tcn.json` | Core-network model configuration |
| `experiments/dataset.py` | Frozen archive/scaler loading and input layout |
| `experiments/loss.py` | Core-network masked level and difference losses |
| `scripts/run_hexinwang_1s_comparison.py` | Masked losses and lagged-difference helper |
| `dataset/hexinwang/build_hexinwang_causal_1s.py` | One-second archive construction |
| `experiments/original_h100_h200/` | Horizon-specific comparison runners |
| `experiments/requested_baselines_original/` | Historical 300-step comparison adapters/runners |

The inspected full architecture is **TCN → FiLM → MoE**. Older **MoE → FiLM** implementations and checkpoints are also present in the working project and must not be silently substituted.

### Environment and installation

The inspected project environment uses Python 3.9.12, PyTorch 2.8.0+cu128, and NumPy 2.0.2. The model interface example runs on CPU; CUDA availability is not required for that example.

With the existing code layout and its dependency file:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Individual baseline runners may require additional source packages and frameworks, including TensorFlow for the inspected TiDE adapter. The root dependency file should not be interpreted as a complete environment specification for every baseline.

### Required data and run artifacts

The inspected core-network loader expects:

```text
dataset/hexinwang/hexinwang_crn_causal_1s.npz
dataset/hexinwang/scaler.npz
```

It validates these SHA-256 values:

```text
dataset: d11936bb1b848d2e0b9371d3662b21660698e19dfdcdf05b51b70b40d969b250
scaler:  eb018aa516129a9b0c10314e8cba0056d8a41c9a32f5c25790965f642d92065f
```

Public download locations for the core-network archive, paper-specific checkpoints, and a complete results bundle are not specified in this document. Describing an interface does not imply that those assets are included in a code-only distribution.

To reproduce a manuscript result, the associated release needs the dataset/split identifiers, preprocessing statistics, model configuration, objective, seed list, training budget, checkpoint-selection rule, checkpoint, and saved predictions/masks. Several historical runners still refer to external local source trees, and the inspected archives use different protocols. They should not be advertised as a single portable command that already reproduces all manuscript tables.

A repository-wide license and final paper citation should be supplied by the authors with the release. Third-party code retains its own attribution and license requirements.

