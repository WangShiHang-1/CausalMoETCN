# CausalMoETCN: Core-Network Dataset

## 1. Dataset overview

The core-network dataset was collected from a Kubernetes-based 5G core-network and radio access network (RAN) testbed built upon free5GC, comprising four base stations and 40 user equipment (UE) devices. Controlled faults and operational events were introduced to record network responses during intervention execution and subsequent recovery. The dataset comprises seven independent collection batches, with telemetry aligned to a one-second time grid (1 Hz). Individual indicators may have different underlying update intervals.

The forecasting task is to predict multivariate KPI trajectories given historical network observations and a prespecified future operational plan. Three throughput indicators are selected as outcomes: N3 uplink bit rate, N3 downlink bit rate, and N6 aggregate bit rate. Eighteen covariates describe session activity, service performance, RAN status, and host resource usage. Operational plans specify the operation types, scheduled start and end times, and target service groups.

| Property | Description |
|---|---|
| Collection environment | Kubernetes-based free5GC core-network and RAN testbed |
| Testbed scale | 4 base stations and 40 UE devices |
| Collection batches | 7 independent batches, D01–D07 |
| Temporal resolution | 1 s |
| Prediction targets | 3 throughput KPIs |
| Observed covariates | 18 indicators |
| Treatment representation | 11-dimensional multi-hot vector: 9 operation types and 2 service-group indicators |
| Operation configurations | 12 joint operation templates and a no-operation baseline |
| Planned intervention duration | 120 or 180 s |
| Historical window | 300 s |
| Forecast horizons | 100, 200, and 300 s |
| Retained forecasting windows | 1,006: 503 intervention windows and 503 matched control windows |

## 2. Indicators and variable roles

The tables below distinguish outcomes (**Y**), historical covariates (**X**), and treatments (**A**), following the notation used in the paper. Historical values of the three outcome KPIs are also provided as inputs. Future covariate observations are not supplied to the forecasting model.

### 2.1 Target KPIs

| Indicator name | Chinese name | Source field | Unit | Role |
|---|---|---|---|---|
| N3 uplink bit rate | N3 上行比特率 | `n3_uplink_bitrate_bps` | bit/s | Y |
| N3 downlink bit rate | N3 下行比特率 | `n3_downlink_bitrate_bps` | bit/s | Y |
| N6 aggregate bit rate | N6 聚合比特率 | `n6_bitrate_bps` | bit/s | Y |

N3 carries user-plane traffic between the RAN and the user plane function (UPF), while N6 connects the UPF to the external data network. The three targets jointly describe traffic responses at these interfaces. Values are stored in bit/s; response plots in the paper display Mbps.

### 2.2 Historical covariates

| Indicator name | Chinese name | Source field | Unit / representation | Role |
|---|---|---|---|---|
| PDU session establishment success rate | PDU 会话建立成功率 | `cp_pdu_session_success_rate` | % | X |
| PDU session establishment failure count | PDU 会话建立失败计数 | `cp_pdu_session_failure_count` | Count | X |
| End-to-end round-trip time | 端到端往返时延 | `end_to_end_rtt_ms` | ms | X |
| Delay jitter | 时延抖动 | `jitter_ms` | ms | X |
| Packet-loss rate | 丢包率 | `packet_loss_rate` | Ratio, 0–1 | X |
| Business completion rate | 业务完成率 | `business_completion_rate` | Ratio, 0–1; 60-s rolling summary | X |
| Active PDU session count | 活跃 PDU 会话数 | `cp_active_session_count` | Count | X |
| Pending PDU session count | 待完成 PDU 会话数 | `cp_pending_session_count` | Count | X |
| PDU session request count | PDU 会话请求计数 | `cp_pdu_session_request_count` | Count | X |
| UE registration success rate | UE 注册成功率 | `registration_success_rate` | % | X |
| UE registration request count | UE 注册请求计数 | `registration_request_count` | Count | X |
| RAN active PDU count | RAN 活跃 PDU 数 | `ran_pdu_active_count` | Count; snapshot | X |
| Ready base-station count | 就绪基站数 | `ready_gnb_count` | Count; snapshot | X |
| Observed UE count | 观测 UE 数 | `observed_ue_count` | Count; snapshot | X |
| N4 bit rate | N4 比特率 | `n4_bitrate_bps` | bit/s | X |
| Host CPU utilization | 宿主机 CPU 利用率 | `host_cpu_pct` | % | X |
| Host memory utilization | 宿主机内存利用率 | `host_mem_used_pct` | % | X |
| Host one-minute load average | 宿主机一分钟平均负载 | `host_loadavg_1` | Dimensionless | X |

Round-trip time, jitter, and packet-loss measurements are telemetry-derived proxies. Count fields are retained in their recorded form; the table does not imply that every count is a per-second event total. Availability and observation-age features accompany the covariates to represent missing or stale observations; these auxiliary features are not additional physical indicators.

## 3. Operational interventions

The treatment vector contains six core-network fault indicators, three RAN operation indicators, and two service-group indicators. Each active plan selects one core-network fault, one RAN operation, and one target service group. Thus, the encoding is multi-hot rather than an 11-class one-hot label.

| Category | Operation / indicator | Chinese description | Encoding field | Role |
|---|---|---|---|---|
| Core network | N3 blocking | N3 链路阻断 | `core_n3_block` | A |
| Core network | N3 delay injection | N3 时延注入 | `core_n3_delay` | A |
| Core network | N3 packet-loss injection | N3 丢包注入 | `core_n3_loss` | A |
| Core network | N4 blocking | N4 链路阻断 | `core_n4_block` | A |
| Core network | N6 blocking | N6 链路阻断 | `core_n6_block` | A |
| Core network | UPF shutdown | UPF 停止 | `core_upf_stop` | A |
| RAN | Base-station restart | 基站重启 | `ran_gnb_restart` | A |
| RAN | UE recreation | UE 重建 | `ran_ue_recreate` | A |
| RAN | UE scale-down | UE 缩容 | `ran_ue_scale_down` | A |
| Service group | Gold group | Gold 业务组 | `tier_gold` | A |
| Service group | Silver group | Silver 业务组 | `tier_silver` | A |

The 12 joint operation templates are listed below. Each cell specifies the RAN operation paired with the core-network fault for the indicated service group.

| Core-network fault | Gold group | Silver group |
|---|---|---|
| N3 blocking | Base-station restart | UE scale-down |
| N3 delay injection | UE scale-down | UE recreation |
| N3 packet-loss injection | UE recreation | Base-station restart |
| N4 blocking | Base-station restart | UE scale-down |
| N6 blocking | UE scale-down | UE recreation |
| UPF shutdown | UE recreation | Base-station restart |

These are the observed joint configurations, rather than all possible combinations of the nine operation types. The no-operation baseline has an all-zero treatment vector. Treatment indicators describe scheduled plans: actual execution timing and lingering effects after the planned end are reflected in the measured KPI trajectories. A planned intervention end does not necessarily mark complete service recovery.

## 4. Forecasting windows and data split

Each intervention window is anchored at the scheduled intervention start and paired with a non-intervention control window. The historical input covers the preceding 300 s. Forecast horizons of 100, 200, and 300 s are evaluated, with the 300-s horizon used for response analysis and ablation studies. The retained dataset contains 503 intervention–control pairs.

Batches are assigned to separate training, validation, and test splits; windows are not randomly redistributed across these splits.

| Split | Collection batches | Intervention windows | Control windows | Total windows |
|---|---|---:|---:|---:|
| Training | D01–D04 | 288 | 288 | 576 |
| Validation | D05 | 72 | 72 | 144 |
| Test | D06–D07 | 143 | 143 | 286 |
| Total | D01–D07 | 503 | 503 | 1,006 |

## 5. Brief preprocessing description

Historical missing values are filled using past observations, with training-derived fallback values where needed. Missing future targets are excluded from the loss and evaluation. Throughput targets undergo a logarithmic transformation relative to a historical baseline, followed by standardization using training-set statistics; validation and test data do not contribute to these statistics. Only historical observations and the prespecified treatment sequence within the forecast horizon are provided as model inputs.
